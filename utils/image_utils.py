import cv2
import numpy as np
from PIL import Image
from typing import Tuple


def load_image_bgr(path: str) -> np.ndarray:
    """
    Load image in BGR format (OpenCV standard).
    Raises error if image is invalid.
    """
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    return img


def load_image_rgb(path: str) -> np.ndarray:
    """
    Load image in RGB format (for PyTorch / transformers).
    """
    bgr = load_image_bgr(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_preserve_aspect(
    image: np.ndarray, max_size: int = 1024
) -> np.ndarray:
    """
    Resize image while preserving aspect ratio.
    Prevents extreme memory usage for very large images.
    """
    h, w = image.shape[:2]

    if max(h, w) <= max_size:
        return image

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert numpy RGB image to PIL Image.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        return Image.fromarray(image)
    raise ValueError("Expected RGB image for PIL conversion")


def normalize_0_1(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0,1] float range.
    """
    return image.astype(np.float32) / 255.0


def compute_blur_variance(gray_image: np.ndarray) -> float:
    """
    Variance of Laplacian — standard blur metric.
    Lower = blurrier.
    """
    lap = cv2.Laplacian(gray_image, cv2.CV_64F)
    return float(lap.var())


def compute_brightness(gray_image: np.ndarray) -> float:
    """
    Compute normalized brightness [0,1].
    """
    return float(gray_image.mean() / 255.0)


def compute_contrast(gray_image: np.ndarray) -> float:
    """
    RMS contrast measure.
    """
    return float(gray_image.std() / 255.0)


def bgr_to_gray(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR to grayscale safely.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def crop_to_square_center(image: np.ndarray) -> np.ndarray:
    """
    Optional utility if needed later for embeddings.
    Crops center square region.
    """
    h, w = image.shape[:2]
    size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return image[y0 : y0 + size, x0 : x0 + size]
