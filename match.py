import numpy as np
from scipy.spatial.distance import cosine

MATCH_THRESHOLD = 0.38
SAMPLE_THRESHOLD = 0.35
MIN_MARGIN = 0.04


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _safe_cosine(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 1.0
    return float(cosine(a, b))


def find_match(test_embedding, suspects):
    test_embedding = _normalize(test_embedding)
    scored_matches = []

    for suspect in suspects:
        centroid = suspect.get("embedding")
        samples = suspect.get("embeddings", [])

        if centroid is None:
            continue

        centroid_distance = _safe_cosine(test_embedding, _normalize(centroid))
        sample_distances = [
            _safe_cosine(test_embedding, _normalize(sample))
            for sample in samples
        ]
        best_sample_distance = min(sample_distances) if sample_distances else centroid_distance
        final_distance = (0.6 * centroid_distance) + (0.4 * best_sample_distance)

        scored_matches.append({
            "name": suspect["name"],
            "distance": final_distance,
            "centroid_distance": centroid_distance,
            "sample_distance": best_sample_distance,
        })

    if not scored_matches:
        return "Unknown", 1.0, None

    scored_matches.sort(key=lambda item: item["distance"])
    best = scored_matches[0]
    second_best_distance = scored_matches[1]["distance"] if len(scored_matches) > 1 else 1.0
    margin = second_best_distance - best["distance"]

    print(
        "Best:",
        best["name"],
        f"distance={best['distance']:.4f}",
        f"sample={best['sample_distance']:.4f}",
        f"margin={margin:.4f}",
    )

    is_strong_match = (
        best["distance"] <= MATCH_THRESHOLD
        and best["sample_distance"] <= SAMPLE_THRESHOLD
        and margin >= MIN_MARGIN
    )

    if not is_strong_match:
        return "Unknown", best["distance"], {
            "margin": margin,
            "best_sample_distance": best["sample_distance"],
            "centroid_distance": best["centroid_distance"],
        }

    return best["name"], best["distance"], {
        "margin": margin,
        "best_sample_distance": best["sample_distance"],
        "centroid_distance": best["centroid_distance"],
    }
