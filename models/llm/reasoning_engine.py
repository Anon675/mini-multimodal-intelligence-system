import json
from typing import Dict, Any

from config.settings import LOCAL_LLM_MODEL
from utils.logger import get_logger


EXPECTED_SCHEMA = {
    "image_quality_score",
    "issues_detected",
    "detected_objects",
    "text_detected",
    "llm_reasoning_summary",
    "final_verdict",
    "confidence",
}


class LocalLLMReasoner:
    """
    Deterministic reasoning layer that:
    - Uses ML features as the primary signal
    - Avoids brittle free-form LLM generation
    - Always returns valid JSON in the required schema
    """

    def __init__(self):
        self.logger = get_logger("llm_reasoning_engine")
        self.logger.info(f"Initialized deterministic reasoner (no text-generation)")

    def _decide_verdict(self, quality_score: float, detected_objects: list) -> str:
        """
        Simple, interpretable decision rule.
        """
        if not detected_objects:
            return "Not suitable"

        if quality_score < 0.5:
            return "Not suitable"

        return "Suitable"

    def _compute_confidence(self, quality_score: float, detected_objects: list) -> float:
        """
        Lightweight confidence scoring.
        """
        base = float(quality_score)

        if not detected_objects:
            return round(base * 0.5, 3)

        return round(min(1.0, base + 0.2), 3)

    def run(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core reasoning method that ALWAYS returns valid JSON.
        """

        quality = float(features.get("image_quality_score", 0.0))
        objects = features.get("detected_objects", [])
        text = features.get("text_detected", [])

        verdict = self._decide_verdict(quality, objects)
        confidence = self._compute_confidence(quality, objects)

        issues = []
        if not objects:
            issues.append("no_reliable_objects_detected")
        if quality < 0.5:
            issues.append("low_image_quality")

        result = {
            "image_quality_score": quality,
            "issues_detected": issues,
            "detected_objects": objects,
            "text_detected": text,
            "llm_reasoning_summary": (
                "Decision based on structured vision signals "
                "(object presence + image quality)."
            ),
            "final_verdict": verdict,
            "confidence": confidence,
        }

        # Guarantee schema completeness
        for key in EXPECTED_SCHEMA:
            result.setdefault(key, None)

        self.logger.info("Returning validated, deterministic reasoning output.")
        return result
