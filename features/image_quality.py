import cv2
import numpy as np
from typing import Dict

def assess_image_quality(image_bgr: np.ndarray) -> Dict[str, float]:
    """
    Compute image quality metrics including:
    - blur variance
    - brightness
    - contrast
    - edge density (NEW: true metric)
    - composite quality score
    """

    # Convert to grayscale once
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ---------- Blur variance ----------
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_variance = float(np.var(laplacian))

    # ---------- Brightness ----------
    brightness = float(np.mean(gray) / 255.0)

    # ---------- Contrast ----------
    contrast = float(np.std(gray) / 255.0)

    # ---------- TRUE EDGE DENSITY (NEW) ----------
    edges = cv2.Canny(gray, 100, 200)

    # Edge density = fraction of edge pixels
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size
    edge_density = float(edge_pixels / total_pixels)

    # ---------- Composite quality score ----------
    # Normalize blur to [0,1] roughly (soft cap)
    blur_norm = min(1.0, blur_variance / 1000.0)

    image_quality_score = float(
        0.4 * blur_norm +
        0.3 * brightness +
        0.2 * contrast +
        0.1 * edge_density
    )

    return {
        "blur_variance": blur_variance,
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "image_quality_score": image_quality_score,
    }
