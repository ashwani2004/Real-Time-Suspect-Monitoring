import os

import numpy as np
from deepface import DeepFace

from mongo import get_suspect_collection

DATASET_PATH = "dataset"
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "opencv"


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _face_area(face_obj):
    area = face_obj.get("facial_area", {})
    return max(0, int(area.get("w", 0))) * max(0, int(area.get("h", 0)))


def _largest_face(faces):
    if not faces:
        return None
    return max(faces, key=_face_area)


def _represent_face(image_source):
    faces = DeepFace.extract_faces(
        img_path=image_source,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
        align=True,
    )
    face = _largest_face(faces)
    if not face:
        return None

    face_img = face.get("face")
    if face_img is None:
        return None

    result = DeepFace.represent(
        img_path=face_img,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
    )
    return _normalize(result[0]["embedding"])


def generate_embeddings():
    collection = get_suspect_collection()
    collection.delete_many({})
    database = []

    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue

        embeddings = []
        processed_images = 0
        skipped_images = 0

        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            if not os.path.isfile(img_path):
                continue

            try:
                emb = _represent_face(img_path)
                if emb is None or np.linalg.norm(emb) == 0:
                    skipped_images += 1
                    print(f"Skipped {img_path}: no usable face detected")
                    continue

                embeddings.append(emb)
                processed_images += 1
            except Exception as e:
                skipped_images += 1
                print(f"Skipped {img_path}: {e}")

        if embeddings:
            avg_emb = _normalize(np.mean(embeddings, axis=0))
            database.append({
                "name": person,
                "embedding": avg_emb.tolist(),
                "embeddings": [emb.tolist() for emb in embeddings],
                "sample_count": len(embeddings),
                "processed_images": processed_images,
                "skipped_images": skipped_images,
                "model_name": MODEL_NAME,
                "is_suspect": True,
            })
            print(
                f"Prepared {person}: kept {processed_images} image(s), skipped {skipped_images}"
            )
        else:
            print(f"Skipped {person}: no valid face embeddings could be created")

    if database:
        collection.insert_many(database)

    print("Embeddings stored in MongoDB")


if __name__ == "__main__":
    generate_embeddings()
