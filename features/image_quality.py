import cv2
import numpy as np
from typing import Dict

from config.settings import (
    BLUR_VARIANCE_THRESHOLD,
    MIN_ACCEPTABLE_BRIGHTNESS,
    MAX_ACCEPTABLE_BRIGHTNESS,
    MIN_CONTRAST,
)
from utils.image_utils import (
    bgr_to_gray,
    compute_blur_variance,
    compute_brightness,
    compute_contrast,
)
from utils.logger import get_logger

logger = get_logger("image_quality")


def assess_image_quality(image_bgr: np.ndarray) -> Dict:
    """
    Compute quantitative image quality signals.
    These are objective ML-style metrics, not LLM opinions.
    """

    gray = bgr_to_gray(image_bgr)

    blur_var = compute_blur_variance(gray)
    brightness = compute_brightness(gray)
    contrast = compute_contrast(gray)

    issues = []

    if blur_var < BLUR_VARIANCE_THRESHOLD:
        issues.append("blurred")

    if brightness < MIN_ACCEPTABLE_BRIGHTNESS:
        issues.append("too_dark")
    elif brightness > MAX_ACCEPTABLE_BRIGHTNESS:
        issues.append("overexposed")

    if contrast < MIN_CONTRAST:
        issues.append("low_contrast")

    # Normalize a simple quality score in [0,1]
    quality_score = float(
        np.clip(
            0.5
            + 0.3 * (blur_var / (BLUR_VARIANCE_THRESHOLD + 1e-6))
            + 0.1 * (1 - abs(brightness - 0.55))
            + 0.1 * contrast,
            0.0,
            1.0,
        )
    )

    result = {
        "blur_variance": float(blur_var),
        "brightness": float(brightness),
        "contrast": float(contrast),
        "issues_detected": issues,
        "image_quality_score": quality_score,
    }

    logger.debug(f"Image quality signals: {result}")
    return result
