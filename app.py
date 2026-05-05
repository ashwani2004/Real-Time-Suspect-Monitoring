from collections import deque
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, Response, jsonify, render_template, send_file
from match import find_match
from mongo import get_log_collection, get_suspect_collection
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "opencv"
COOLDOWN = 10
FRAME_SKIP = 2
CONSECUTIVE_MATCHES_REQUIRED = 3
LOG_LIMIT = 12
app = Flask(__name__)
class LocalMonitor:
    def __init__(self):
        self.suspect_collection = get_suspect_collection()
        self.log_collection = get_log_collection()
        self.reload_suspects()
        self.last_detected = {}
        self.recent_predictions = deque(maxlen=CONSECUTIVE_MATCHES_REQUIRED)
        self.frame_count = 0
        self.last_status = {
            "label": "Idle",
            "color": (0, 255, 255),
            "box": None,
        }
    def reload_suspects(self):
        self.suspects = list(self.suspect_collection.find())

    def normalize(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def face_area(self, face_obj):
        area = face_obj.get("facial_area", {})
        return max(0, int(area.get("w", 0))) * max(0, int(area.get("h", 0)))

    def largest_face(self, faces):
        if not faces:
            return None
        return max(faces, key=self.face_area)

    def get_box(self, face_obj, frame_shape):
        area = face_obj.get("facial_area", {})
        frame_h, frame_w = frame_shape[:2]
        x = max(0, int(area.get("x", 0)))
        y = max(0, int(area.get("y", 0)))
        w = max(0, int(area.get("w", 0)))
        h = max(0, int(area.get("h", 0)))
        x2 = min(frame_w - 1, x + w)
        y2 = min(frame_h - 1, y + h)
        return x, y, x2, y2

    def analyze_frame(self, frame):
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )
        face = self.largest_face(faces)
        if not face:
            return None, None

        face_img = face.get("face")
        if face_img is None:
            return None, None

        result = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
        )
        embedding = self.normalize(result[0]["embedding"])
        box = self.get_box(face, frame.shape)
        return embedding, box

    def process_frame(self, frame):
        self.frame_count += 1
        label = self.last_status["label"]
        color = self.last_status["color"]
        box = self.last_status["box"]

        try:
            if self.frame_count % FRAME_SKIP == 0 and self.suspects:
                emb, detected_box = self.analyze_frame(frame)
                box = detected_box

                if emb is None or np.linalg.norm(emb) == 0:
                    self.recent_predictions.clear()
                    label = "No face detected"
                    color = (0, 255, 255)
                else:
                    name, dist, match_meta = find_match(emb, self.suspects)
                    confidence = max(0.0, 1.0 - dist)

                    if name == "Unknown":
                        self.recent_predictions.clear()
                        label = f"Unknown ({confidence:.2f})"
                        color = (0, 0, 255)
                    else:
                        self.recent_predictions.append(name)
                        stable_match = (
                            len(self.recent_predictions) == CONSECUTIVE_MATCHES_REQUIRED
                            and len(set(self.recent_predictions)) == 1
                        )

                        if stable_match:
                            label = f"{name} ({confidence:.2f})"
                            color = (40, 200, 120)
                            current_time = datetime.now().timestamp()

                            if name not in self.last_detected or current_time - self.last_detected[name] > COOLDOWN:
                                image_name = f"alert_{name}_{int(current_time)}.jpg"
                                cv2.imwrite(str(BASE_DIR / image_name), frame)
                                self.log_collection.insert_one({
                                    "name": name,
                                    "timestamp": datetime.now(),
                                    "confidence": float(confidence),
                                    "status": "detected",
                                    "image": image_name,
                                    "model_name": MODEL_NAME,
                                    "match_distance": float(dist),
                                    "margin": float(match_meta["margin"]) if match_meta else None,
                                })
                                self.last_detected[name] = current_time
                        else:
                            label = (
                                f"Verifying {name} "
                                f"({len(self.recent_predictions)}/{CONSECUTIVE_MATCHES_REQUIRED})"
                            )
                            color = (0, 215, 255)
            elif not self.suspects:
                label = "No suspects loaded"
                color = (0, 165, 255)
        except Exception as exc:
            self.recent_predictions.clear()
            label = "Detection issue"
            color = (0, 165, 255)
            print("Dashboard detection error:", exc)

        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            label,
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

        self.last_status = {"label": label, "color": color, "box": box}
        return frame

    def stream_frames(self):
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                processed = self.process_frame(frame)
                encoded_ok, buffer = cv2.imencode(".jpg", processed)
                if not encoded_ok:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
        finally:
            cap.release()


monitor = LocalMonitor()


def _first_dataset_image(name):
    person_dir = DATASET_DIR / name
    if not person_dir.exists():
        return None

    files = sorted(
        path for path in person_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return files[0] if files else None


def _serialize_log(document):
    timestamp = document.get("timestamp")
    if hasattr(timestamp, "isoformat"):
        timestamp = timestamp.isoformat(timespec="seconds")
    else:
        timestamp = str(timestamp)

    image_name = document.get("image")
    image_url = f"/alert_image/{image_name}" if image_name else None
    return {
        "name": document.get("name", "Unknown"),
        "confidence": round(float(document.get("confidence", 0.0)), 3),
        "status": document.get("status", "detected"),
        "timestamp": timestamp,
        "image_url": image_url,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        monitor.stream_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stats")
def stats():
    monitor.reload_suspects()
    suspect_count = len(monitor.suspects)
    log_count = monitor.log_collection.count_documents({})
    latest = monitor.log_collection.find_one(sort=[("timestamp", -1)])
    return jsonify({
        "suspect_count": suspect_count,
        "log_count": log_count,
        "latest_detection": _serialize_log(latest) if latest else None,
        "status": monitor.last_status["label"],
    })


@app.route("/api/logs")
def logs():
    cursor = monitor.log_collection.find().sort("timestamp", -1).limit(LOG_LIMIT)
    return jsonify([_serialize_log(document) for document in cursor])


@app.route("/api/suspects")
def suspects():
    monitor.reload_suspects()
    payload = []
    for suspect in sorted(monitor.suspects, key=lambda item: item.get("name", "")):
        image_path = _first_dataset_image(suspect.get("name", ""))
        payload.append({
            "name": suspect.get("name", "Unknown"),
            "sample_count": suspect.get("sample_count", 0),
            "processed_images": suspect.get("processed_images", 0),
            "skipped_images": suspect.get("skipped_images", 0),
            "model_name": suspect.get("model_name", MODEL_NAME),
            "image_url": f"/suspect_image/{suspect.get('name')}" if image_path else None,
        })
    return jsonify(payload)


@app.route("/suspect_image/<name>")
def suspect_image(name):
    image_path = _first_dataset_image(name)
    if image_path is None:
        return ("Not found", 404)
    return send_file(image_path)


@app.route("/alert_image/<path:filename>")
def alert_image(filename):
    safe_path = (BASE_DIR / filename).resolve()
    if safe_path.parent != BASE_DIR.resolve() or not safe_path.exists():
        return ("Not found", 404)
    return send_file(safe_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
