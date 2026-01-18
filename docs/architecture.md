# System Architecture — Mini Multimodal Intelligence System

## High-Level Design

The system follows a **signal-first, LLM-second** architecture with four clear layers:

1. **Vision Feature Extraction Layer (Pre-LLM Intelligence)**
2. **Feature Processing Layer**
3. **Local LLM Reasoning Layer**
4. **Inference Orchestration Layer**

This separation ensures that the LLM is used for reasoning, not perception.

---

## 1) Vision Feature Extraction Layer

Three independent vision subsystems operate in parallel:

### a) Object Detection — YOLOv8n
- Model: `yolov8n.pt` (pretrained)
- Role:
  - Identify objects present in the scene
  - Provide high-level semantic context (e.g., “shoe”, “person”, “table”)
- Output:
  - List of detected objects (deduplicated labels only)
- Rationale:
  - Fast
  - Reliable
  - No training required
  - Strong baseline for real-world images

### b) Image Embeddings — CLIP (ViT-B/32)
- Model: `openai/clip-vit-base-patch32`
- Role:
  - Capture high-level visual semantics
  - Represent image meaning in a dense vector space
- Use in this system:
  - Not directly used for decision-making
  - Stored for explainability, future retrieval, or similarity search
- Rationale:
  - Industry-standard multimodal representation
  - Bridges vision and language spaces

### c) OCR — Tesseract
- Extracts any readable text from the image
- Cleans and normalizes results
- Important for:
  - Detecting logos, branding, or watermarks
  - Identifying possible trust or professionalism signals

---

## 2) Feature Processing Layer

Raw model outputs are converted into **structured, interpretable signals**:

- **Object summarization**
  - Removes duplicates
  - Keeps only semantic labels

- **Text summarization**
  - Removes noise
  - Normalizes case
  - Filters meaningless fragments

- **Image quality analysis**
  - Blur variance (Laplacian)
  - Brightness (normalized mean)
  - Contrast (RMS)
  - Converts numeric metrics into qualitative issues like:
    - “blurred”
    - “too_dark”
    - “low_contrast”

This ensures that the LLM reasons over **measured evidence, not pixels.**

---

## 3) Local LLM Reasoning Layer

Model: `google/flan-t5-small` (local, CPU-friendly)

The LLM:
- Does **not** see the image
- Receives only structured signals:
  - Detected objects
  - OCR text
  - Blur score
  - Brightness
  - Contrast

Tasks performed by the LLM:
1. Judge suitability for professional e-commerce use  
2. List concrete issues  
3. Provide a short reasoning summary  
4. Return a verdict: `"Suitable"` or `"Not suitable"`  
5. Provide confidence score (0–1)

Output format is strict JSON to ensure machine usability.

---

## 4) Inference Pipeline (Orchestration Layer)

The pipeline:
1. Loads image
2. Runs all vision models
3. Computes quality metrics
4. Sends structured features to LLM
5. Merges ML signals + LLM reasoning into final schema:
   - `image_quality_score`
   - `issues_detected`
   - `detected_objects`
   - `text_detected`
   - `llm_reasoning_summary`
   - `final_verdict`
   - `confidence`

Each image produces one standalone JSON file.

---

## Data Flow Diagram

