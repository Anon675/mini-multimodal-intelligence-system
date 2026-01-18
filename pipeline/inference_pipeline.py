from typing import Dict, Any

from config.settings import OUTPUT_KEYS, YOLO_CONFIDENCE_THRESHOLD
from utils.image_utils import load_image_bgr
from utils.logger import get_logger

from models.vision.yolo_detector import YoloObjectDetector
from models.vision.clip_embedder import ClipImageEmbedder
from models.vision.ocr_extractor import OCRExtractor
from models.llm.reasoning_engine import LocalLLMReasoner

from features.object_features import summarize_objects
from features.text_features import summarize_text
from features.image_quality import assess_image_quality


logger = get_logger("inference_pipeline")


class InferencePipeline:
    """
    End-to-end pipeline:
    1) Load image
    2) Extract vision features
    3) Gate detections by confidence ONLY (no class filter)
    4) Pass structured features to reasoning layer
    5) Merge into final schema
    """

    def __init__(self):
        logger.info("Initializing inference pipeline...")

        self.detector = YoloObjectDetector()
        self.embedder = ClipImageEmbedder()
        self.ocr = OCRExtractor()
        self.llm = LocalLLMReasoner()

        logger.info("Pipeline ready.")

    def _assemble_features(self, image_path: str) -> Dict[str, Any]:
        logger.info(f"Extracting features for: {image_path}")

        image_bgr = load_image_bgr(image_path)

        # -------- OBJECT DETECTION (CONFIDENCE GATE ONLY) --------
        detections = self.detector.detect(image_bgr)

        filtered_detections = [
            d for d in detections
            if float(d.get("confidence", 0.0)) >= YOLO_CONFIDENCE_THRESHOLD
        ]

        detected_objects = summarize_objects(filtered_detections)

        # -------- OCR (no early exit) --------
        raw_text = self.ocr.extract_text(image_path)
        text_detected = summarize_text(raw_text)

        # -------- IMAGE QUALITY --------
        quality = assess_image_quality(image_bgr)

        # -------- EMBEDDING (kept for completeness) --------
        embedding = self.embedder.embed_image(image_path)

        features = {
            "detected_objects": detected_objects,
            "text_detected": text_detected,
            "blur_variance": quality["blur_variance"],
            "brightness": quality["brightness"],
            "contrast": quality["contrast"],
            "image_quality_score": quality["image_quality_score"],
            "vision_embedding": embedding,
        }

        logger.info("Feature extraction complete.")
        return features

    def _merge_outputs(
        self,
        features: Dict[str, Any],
        llm_output: Dict[str, Any],
    ) -> Dict[str, Any]:

        result = {
            "image_quality_score": float(
                features.get("image_quality_score", 0.0)
            ),
            "issues_detected": list(
                llm_output.get("issues_detected", [])
            ),
            "detected_objects": features.get("detected_objects", []),
            "text_detected": features.get("text_detected", []),
            "llm_reasoning_summary": llm_output.get(
                "llm_reasoning_summary",
                "No reasoning available.",
            ),
            "final_verdict": llm_output.get(
                "final_verdict",
                "Not suitable",
            ),
            "confidence": float(
                llm_output.get("confidence", 0.5)
            ),
        }

        for key in OUTPUT_KEYS:
            result.setdefault(key, None)

        return result

    def run(self, image_path: str) -> Dict[str, Any]:
        logger.info(f"Running pipeline on: {image_path}")

        features = self._assemble_features(image_path)
        llm_output = self.llm.run(features)
        final_result = self._merge_outputs(features, llm_output)

        logger.info("Pipeline complete.")
        return final_result
