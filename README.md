# Mini Multimodal Intelligence System

A small but conceptually deep **multimodal ML pipeline** that evaluates whether an image is suitable for professional e-commerce use.

This project is not “send image to GPT.”  
It is a **three-stage system**:

1. **Perception layer (real ML):** Computer vision extracts measurable evidence from the image.
2. **Reasoning layer (local LLM):** A lightweight LLM reasons over structured features.
3. **Validation layer:** The system checks LLM reliability and falls back safely if it fails.

The goal was to demonstrate **how ML engineers should build systems — not prompts.**

---

## 🎯 What this system actually does

For each input image, the pipeline:

1. **Loads the image once**
2. Runs **YOLOv8** for object detection  
3. Runs **Tesseract OCR** for text extraction  
4. Computes **image quality metrics**:
   - Blur variance  
   - Brightness  
   - Contrast  
   - Composite quality score  
5. Extracts a **CLIP embedding** (for explainability + future retrieval use)
6. Sends **structured features (not the image)** to a **local LLM**
7. Validates the LLM’s JSON output strictly  
8. Produces a final structured verdict


🧠 Core design principles (why this is not a toy project)
1) Vision-first intelligence
 -The system makes real ML judgments before calling any LLM.
 -This avoids the common failure mode where people:

   “Just send the image to GPT and hope.”

      Here, the LLM only reasons over measured signals, not pixels.

2) LLM is a reasoning assistant, not a brain
The local LLM:

-Never sees the image

-Never replaces vision models

-Only interprets structured evidence

-This matches how production systems are actually built.

3) Lightweight evaluation & reliability layer
-Instead of trusting the LLM blindly, the system enforces:

-Required keys

-Correct data types

-Valid numeric ranges

-Allowed verdict values

-If the LLM returns bad JSON → the system:

-Logs the failure

-Flags "llm_format_error"

-Falls back to deterministic vision-based judgment

-This is intentional engineering robustness, not a bug.

🧩 Why a local LLM (not OpenAI API)?
-This was a deliberate trade-off, not ignorance.

Reason 1 — Budget constraints
-Sending thousands of images to a paid API is expensive and unrealistic for many startups and research labs. A local model keeps costs predictable and controllable.

Reason 2 — Privacy & compliance
-Many companies cannot send images to external cloud APIs due to legal and security policies. A local model is safer and enterprise-friendly.

Reason 3 — System realism
-Real production pipelines often:

Run vision on-device or on-prem

-Use local reasoning models

-Avoid dependence on third-party APIs

-This project reflects that reality.

Trade-off acknowledged
-A small local LLM is weaker at:

-Strict JSON formatting

-Complex reasoning

-Instead of pretending this isn’t true, the system detects and handles it explicitly.


▶️ How to run
1) Install dependencies
pip install -r requirements.txt
2) Run the pipeline
python run.py \
  --input "path/to/data/inputs" \
  --output "path/to/data/outputs"
Each image produces a .json file in the outputs folder.

🚀 How this would scale in production
If this were deployed at scale, I would add:

GPU acceleration for YOLO + CLIP

Batch processing

FastAPI microservice wrapper

Automated evaluation on labeled dataset

A/B comparison between different LLMs

Caching for repeated images

✅ What this project demonstrates
This project shows that I can:

Build real ML pipelines, not just prompts

Combine vision + LLM intelligently

Handle model uncertainty honestly

Write clean, modular, readable code

Think like an ML engineer, not a chatbot user

### Final JSON output example
```json
{
  "image_quality_score": 0.63,
  "issues_detected": ["llm_format_error"],
  "detected_objects": ["laptop", "cup"],
  "text_detected": [],
  "llm_reasoning_summary": "LLM output was malformed. Decision based solely on vision quality metrics.",
  "final_verdict": "Suitable",
  "confidence": 0.25
}
