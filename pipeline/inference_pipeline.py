import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from config.settings import OUTPUT_KEYS
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
    End-to-end pipeline that:
    1) Loads image
    2) Extracts ML features (objects, embeddings, OCR, quality)
    3) Sends structured signals to local LLM for reasoning
    4) Returns structured JSON output
    """

    def __init__(self):
        logger.info("Initializing inference pipeline...")

        # Vision models (pre-LLM intelligence)
        self.detector = YoloObjectDetector()
        self.embedder = ClipImageEmbedder()
        self.ocr = OCRExtractor()

        # Local reasoning model
        self.llm = LocalLLMReasoner()

        logger.info("Pipeline ready.")

    def _assemble_features(self, image_path: str) -> Dict[str, Any]:
        """
        Extract all signals BEFORE calling the LLM.
        This is the core ML part of the system.
        """

        logger.info(f"Extracting features for: {image_path}")

        # Load image once
        image_bgr = load_image_bgr(image_path)

        # --- 1) Object detection ---
        detections = self.detector.detect(image_bgr)
        detected_objects = summarize_objects(detections)

        # --- 2) OCR text extraction ---
        raw_text = self.ocr.extract_text(image_path)
        text_detected = summarize_text(raw_text)

        # --- 3) Image quality metrics ---
        quality = assess_image_quality(image_bgr)

        # --- 4) Image embeddings (for future use / explainability) ---
        embedding = self.embedder.embed_image(image_path)

        features = {
            "detected_objects": detected_objects,
            "text_detected": text_detected,
            "blur_variance": quality["blur_variance"],
            "brightness": quality["brightness"],
            "contrast": quality["contrast"],
            "image_quality_score": quality["image_quality_score"],
            "vision_embedding": embedding,  # kept for completeness
        }

        logger.info("Feature extraction complete.")
        return features

    def _merge_outputs(self, features: Dict[str, Any], llm_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine ML signals + LLM reasoning into final structured schema.
        """

        result = {
            "image_quality_score": float(features["image_quality_score"]),
            "issues_detected": list(set(
                features.get("issues_detected", []) + llm_output.get("issues_detected", [])
            )),
            "detected_objects": features["detected_objects"],
            "text_detected": features["text_detected"],
            "llm_reasoning_summary": llm_output.get(
                "llm_reasoning_summary",
                "No reasoning available."
            ),
            "final_verdict": llm_output.get("final_verdict", "Not suitable"),
            "confidence": float(llm_output.get("confidence", 0.5)),
        }

        # Ensure output matches required keys
        for key in OUTPUT_KEYS:
            if key not in result:
                result[key] = None

        return result

    def run(self, image_path: str) -> Dict[str, Any]:
        """
        Public method called by run.py.
        Returns final JSON-ready dictionary.
        """

        logger.info(f"Running pipeline on: {image_path}")

        # Step 1: Extract features (ML judgment)
        features = self._assemble_features(image_path)

        # Step 2: LLM reasoning over structured signals
        llm_output = self.llm.reason(features)

        # Step 3: Merge into final structured output
        final_result = self._merge_outputs(features, llm_output)

        logger.info("Pipeline complete.")
        return final_result
