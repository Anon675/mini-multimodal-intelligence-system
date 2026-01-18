import os
from pathlib import Path

# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
LOG_LEVEL = "INFO"

# ---------------------------------------------------------
# VISION MODELS
# ---------------------------------------------------------

# Stronger detector (your deliberate design choice)
YOLO_MODEL_NAME = "yolov8m.pt"

# Confidence gate (Fix-1)
YOLO_CONFIDENCE_THRESHOLD = 0.45

# IoU threshold for Non-Max Suppression (required by your detector)
YOLO_IOU_THRESHOLD = 0.45

# Backward-compatible alias (because your detector imports this name)
YOLO_CONF_THRESHOLD = YOLO_CONFIDENCE_THRESHOLD

# Device control used by yolo_detector.py
USE_GPU_IF_AVAILABLE = False

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ---------------------------------------------------------
# LOCAL LLM
# ---------------------------------------------------------
LOCAL_LLM_MODEL = "google/flan-t5-base"

# ---------------------------------------------------------
# OUTPUT SCHEMA (single source of truth)
# ---------------------------------------------------------
OUTPUT_KEYS = {
    "image_quality_score",
    "issues_detected",
    "detected_objects",
    "text_detected",
    "llm_reasoning_summary",
    "final_verdict",
    "confidence",
}

# ---------------------------------------------------------
# OCR SETTINGS
# ---------------------------------------------------------
TESSERACT_LANG = "eng"
TESSERACT_TIMEOUT_SEC = 10

# ---------------------------------------------------------
# IMAGE QUALITY DEFAULTS
# ---------------------------------------------------------
DEFAULT_BLUR_THRESHOLD = 100.0
DEFAULT_BRIGHTNESS_MIN = 50
DEFAULT_BRIGHTNESS_MAX = 200

# ---------------------------------------------------------
# HUGGINGFACE CACHE
# ---------------------------------------------------------
HF_CACHE_DIR = os.getenv(
    "HF_HOME",
    str(BASE_DIR / ".hf_cache")
)
