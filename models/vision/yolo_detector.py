import torch
from ultralytics import YOLO
from typing import List, Dict

from config.settings import (
    YOLO_MODEL_NAME,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    USE_GPU_IF_AVAILABLE,
)
from utils.logger import get_logger

logger = get_logger("yolo_detector")


class YoloObjectDetector:
    def __init__(self):
        logger.info(f"Loading YOLO model: {YOLO_MODEL_NAME}")

        self.device = "cuda" if (USE_GPU_IF_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.model = YOLO(YOLO_MODEL_NAME)

        # Move model to device if supported
        try:
            self.model.to(self.device)
        except Exception:
            logger.warning("Could not explicitly move YOLO to device; continuing with default.")

        logger.info(f"YOLO running on: {self.device}")

    def detect(self, image_bgr) -> List[Dict]:
        """
        Run object detection and return structured results.
        Each object contains: name, confidence, bbox.
        """

        results = self.model.predict(
            source=image_bgr,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            verbose=False,
        )

        detections = []

        if len(results) == 0:
            return detections

        r = results[0]

        boxes = r.boxes
        if boxes is None:
            return detections

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = map(float, boxes.xyxy[i])

            detections.append({
                "label": r.names[cls_id],
                "confidence": round(conf, 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                }
            })

        logger.debug(f"Detected {len(detections)} objects")
        return detections
