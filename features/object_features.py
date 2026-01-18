from typing import List, Dict

from utils.logger import get_logger

logger = get_logger("object_features")


def summarize_objects(detections: List[Dict]) -> List[str]:
    """
    Convert raw YOLO detections into a clean list of unique object names.
    Keeps only high-level semantic labels (no boxes, no scores).
    """

    if not detections:
        return []

    labels = [d["label"] for d in detections if "label" in d]

    # Deduplicate while preserving order
    seen = set()
    unique_labels = []
    for lbl in labels:
        if lbl not in seen:
            seen.add(lbl)
            unique_labels.append(lbl)

    logger.debug(f"Summarized objects: {unique_labels}")
    return unique_labels
