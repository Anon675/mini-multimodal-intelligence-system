import json
import torch
from typing import Dict, Any

from transformers import T5ForConditionalGeneration, T5Tokenizer

from config.settings import (
    LOCAL_LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    USE_GPU_IF_AVAILABLE,
)
from utils.logger import get_logger

logger = get_logger("llm_reasoning_engine")


class LocalLLMReasoner:
    """
    Local LLM reasoning layer.
    IMPORTANT: This model does NOT see the image.
    It only reasons over extracted features.
    """

    def __init__(self):
        logger.info(f"Loading local LLM: {LOCAL_LLM_MODEL}")

        self.device = "cuda" if (USE_GPU_IF_AVAILABLE and torch.cuda.is_available()) else "cpu"

        self.tokenizer = T5Tokenizer.from_pretrained(LOCAL_LLM_MODEL)
        self.model = T5ForConditionalGeneration.from_pretrained(LOCAL_LLM_MODEL)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Local LLM running on: {self.device}")

    def _build_prompt(self, features: Dict[str, Any]) -> str:
        """
        Convert structured vision features into a reasoning prompt.
        This is the ONLY place where the LLM interacts with the data.
        """

        prompt = f"""
You are an image quality and trustworthiness assessor for e-commerce images.

You will NOT see the image.
You must reason ONLY from the structured signals below.

Detected objects: {features.get("detected_objects", [])}
OCR text: {features.get("text_detected", [])}
Blur variance: {features.get("blur_variance")}
Brightness: {features.get("brightness")}
Contrast: {features.get("contrast")}

Task:
1) Decide whether the image is suitable for a professional e-commerce product listing.
2) List concrete issues if any.
3) Provide a short reasoning summary.
4) Give a final verdict: "Suitable" or "Not suitable".
5) Estimate confidence between 0 and 1.

Respond STRICTLY in valid JSON with this exact structure:

{{
  "issues_detected": [list of strings],
  "llm_reasoning_summary": "short paragraph",
  "final_verdict": "Suitable or Not suitable",
  "confidence": float between 0 and 1
}}
"""
        return prompt.strip()

    @torch.no_grad()
    def reason(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run local LLM reasoning over extracted features.
        Returns parsed JSON.
        """

        prompt = self._build_prompt(features)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            do_sample=False,
        )

        response_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Try to parse JSON safely
        try:
            result = json.loads(response_text)
        except Exception:
            logger.warning("LLM returned malformed JSON — applying safe fallback.")

            result = {
                "issues_detected": [],
                "llm_reasoning_summary": response_text,
                "final_verdict": "Not suitable",
                "confidence": 0.5,
            }

        return result
