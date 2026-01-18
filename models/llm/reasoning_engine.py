import json
import re
from typing import Dict, Any

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config.settings import LOCAL_LLM_MODEL
from utils.logger import get_logger


# ---------------------------------------------------------
# SINGLE SOURCE OF TRUTH FOR OUTPUT FORMAT
# ---------------------------------------------------------
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
    Local LLM reasoning engine with:
    - structured prompting
    - robust JSON recovery
    - strict schema validation
    - safe fallback grounded in vision features
    """

    def __init__(self):
        self.logger = get_logger("llm_reasoning_engine")

        self.logger.info(f"Loading local LLM: {LOCAL_LLM_MODEL}")

        self.tokenizer = T5Tokenizer.from_pretrained(
            LOCAL_LLM_MODEL,
            legacy=True,
        )

        self.model = T5ForConditionalGeneration.from_pretrained(
            LOCAL_LLM_MODEL
        )

        self.device = "cpu"
        self.model.to(self.device)

        self.logger.info(f"LLM loaded on: {self.device}")

    # ---------------------------------------------------------
    # PROMPT
    # ---------------------------------------------------------
    def _build_prompt(self, features: Dict[str, Any]) -> str:
        return f"""
You are a structured reasoning assistant.

Judge whether the image is suitable for professional e-commerce use
based ONLY on the following structured evidence.

EVIDENCE:
- detected_objects: {features.get("detected_objects", [])}
- text_detected: {features.get("text_detected", [])}
- blur_score: {features.get("blur_variance", 0.0)}
- brightness: {features.get("brightness", 0.0)}
- contrast: {features.get("contrast", 0.0)}

YOU MUST RETURN VALID JSON ONLY — no explanations, no prose.

Required keys (exactly these names):
{list(EXPECTED_SCHEMA)}

Example format:

{{
  "image_quality_score": 0.72,
  "issues_detected": ["blurred", "too_dark"],
  "detected_objects": ["shoe"],
  "text_detected": [],
  "llm_reasoning_summary": "Image is slightly blurry and dark.",
  "final_verdict": "Not suitable",
  "confidence": 0.81
}}
"""

    # ---------------------------------------------------------
    # LIGHTWEIGHT VALIDATION LAYER
    # ---------------------------------------------------------
    def validate_llm_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        missing = EXPECTED_SCHEMA - set(output.keys())
        if missing:
            raise ValueError(f"LLM output missing fields: {missing}")

        if not isinstance(output["confidence"], (int, float)):
            raise ValueError("confidence must be numeric")

        if not isinstance(output["image_quality_score"], (int, float)):
            raise ValueError("image_quality_score must be numeric")

        if not isinstance(output["issues_detected"], list):
            raise ValueError("issues_detected must be a list")

        if not isinstance(output["detected_objects"], list):
            raise ValueError("detected_objects must be a list")

        if not isinstance(output["text_detected"], list):
            raise ValueError("text_detected must be a list")

        if output["final_verdict"] not in ["Suitable", "Not suitable"]:
            raise ValueError(
                "final_verdict must be 'Suitable' or 'Not suitable'"
            )

        if not (0.0 <= float(output["confidence"]) <= 1.0):
            raise ValueError("confidence must be between 0 and 1")

        if not (0.0 <= float(output["image_quality_score"]) <= 1.0):
            raise ValueError("image_quality_score must be between 0 and 1")

        output["confidence"] = float(output["confidence"])
        output["image_quality_score"] = float(output["image_quality_score"])

        return output

    # ---------------------------------------------------------
    # SAFE FALLBACK GROUNDED IN VISION FEATURES
    # ---------------------------------------------------------
    def _vision_fallback(self, features: Dict[str, Any]) -> Dict[str, Any]:
        score = float(features.get("image_quality_score", 0.5))

        return {
            "image_quality_score": score,
            "issues_detected": ["llm_format_error"],
            "detected_objects": features.get("detected_objects", []),
            "text_detected": features.get("text_detected", []),
            "llm_reasoning_summary": (
                "LLM output was malformed. Decision based solely on "
                "vision quality metrics."
            ),
            "final_verdict": "Suitable" if score > 0.6 else "Not suitable",
            "confidence": 0.25,
        }

    # ---------------------------------------------------------
    # MAIN INFERENCE METHOD
    # ---------------------------------------------------------
    def run(self, features: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(features)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=2,
                early_stopping=True,
            )

        raw_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()

        self.logger.debug(f"Raw LLM output:\n{raw_text}")

        # -------- TRY TO EXTRACT VALID JSON BLOCK --------
        matches = re.findall(r"\{.*?\}", raw_text, re.DOTALL)

        if matches:
            candidate = matches[0]
            try:
                parsed = json.loads(candidate)
                return self.validate_llm_output(parsed)
            except Exception:
                self.logger.warning(
                    f"Found JSON-like block but invalid:\n{candidate}"
                )

        # -------- FALLBACK IF LLM FAILS --------
        self.logger.error(
            "LLM failed to return valid JSON — using structured fallback."
        )

        return self._vision_fallback(features)
