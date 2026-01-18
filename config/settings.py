import os
from pathlib import Path

# -----------------------------
# PROJECT ROOT
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# -----------------------------
# DATA PATHS
# -----------------------------
INPUT_DIR = BASE_DIR / "data" / "inputs"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

# Ensure output directory exists at runtime
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# VISION MODEL SETTINGS
# -----------------------------

# YOLO object detection
YOLO_MODEL_NAME = "yolov8n.pt"   # small, fast, reliable baseline
YOLO_CONF_THRESHOLD = 0.25       # balance between precision and recall
YOLO_IOU_THRESHOLD = 0.45

# CLIP image embedding model (local)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# OCR settings
TESSERACT_LANG = "eng"

# -----------------------------
# IMAGE QUALITY SETTINGS
# -----------------------------

# Blur detection
BLUR_VARIANCE_THRESHOLD = 100.0   # lower = blurrier

# Brightness range (normalized 0-1)
MIN_ACCEPTABLE_BRIGHTNESS = 0.25
MAX_ACCEPTABLE_BRIGHTNESS = 0.85

# Contrast threshold
MIN_CONTRAST = 0.20

# -----------------------------
# LOCAL LLM SETTINGS (NO OPENAI)
# -----------------------------

# We will use a lightweight local reasoning model
LOCAL_LLM_MODEL = "google/flan-t5-base"

# Generation parameters
LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0.2  # low randomness = more reliable output

# -----------------------------
# OUTPUT SCHEMA KEYS
# -----------------------------
OUTPUT_KEYS = [
    "image_quality_score",
    "issues_detected",
    "detected_objects",
    "text_detected",
    "llm_reasoning_summary",
    "final_verdict",
    "confidence",
]

# -----------------------------
# SCALABILITY DEFAULTS
# -----------------------------
BATCH_SIZE = 1           # keep simple for this task
USE_GPU_IF_AVAILABLE = True

# -----------------------------
# LOGGING
# -----------------------------
LOG_LEVEL = "INFO"

# -----------------------------
# HELPER FLAGS
# -----------------------------
ALLOW_CACHING = False     # optional future improvement
ENABLE_EXPLAINABILITY = True
