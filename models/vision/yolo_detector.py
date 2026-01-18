from typing import List, Dict, Any

import torch
import numpy as np
from ultralytics import YOLO

from config.settings import (
    YOLO_MODEL_NAME,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    USE_GPU_IF_AVAILABLE,
)
from utils.logger import get_logger


class YoloObjectDetector:
    """
    Robust, class-agnostic object detector.

    Design principles:
    - No hardcoded allowed classes
    - No manual relabeling
    - Trust YOLO's native taxonomy
    - Only gate by confidence + IoU
    """

    def __init__(self):
        self.logger = get_logger("yolo_detector")

        self.logger.info(f"Loading YOLO model: {YOLO_MODEL_NAME}")
        self.model = YOLO(YOLO_MODEL_NAME)

        # Device selection
        if USE_GPU_IF_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model.to(self.device)
        self.logger.info(f"YOLO running on: {self.device}")

        # Cache class names from the model itself
        self.class_names = getattr(self.model, "names", {})

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLO detection and return native labels.
        No manual class constraints.
        """

        results = self.model.predict(
            image_bgr,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            verbose=False,
            device=self.device,
        )[0]

        detections: List[Dict[str, Any]] = []

        if not hasattr(results, "boxes") or results.boxes is None:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()

        for box, score, cid in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)

            label = self.class_names.get(int(cid), f"class_{int(cid)}")

            detections.append(
                {
                    "label": label,
                    "confidence": float(score),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return detections
