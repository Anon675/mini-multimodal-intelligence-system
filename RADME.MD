
The LLM **never sees the image**. It reasons only over extracted features, which reduces hallucination risk and increases interpretability.

---

## System Architecture

### 1) Vision Feature Extraction Layer (Pre-LLM)

| Component | Model | Role in System |
|-----------|--------|----------------|
| Object Detection | **YOLOv8n** | Detects semantic entities in the scene (e.g., “shoe”, “person”, “table”). Provides context rather than final decisions. |
| Image Embeddings | **CLIP ViT-B/32** | Encodes high-level visual semantics into a dense vector. Stored for explainability and potential retrieval use. |
| OCR | **Tesseract** | Extracts visible text, logos, or branding that may influence professionalism or trust. |
| Quality Metrics | Custom OpenCV pipeline | Measures blur, brightness, and contrast to quantify technical image suitability. |

These signals are **ground truth measurements**, not LLM interpretations.

---

### 2) Feature Processing Layer  

Raw model outputs are normalized and summarized into clean signals:

- **Object summarization:** deduplicated labels only  
- **Text normalization:** noise removal and case normalization  
- **Image quality assessment:**  
  - Variance of Laplacian → blur  
  - Mean intensity → brightness  
  - Standard deviation → contrast  

The system converts numeric values into interpretable issues such as:
- `"blurred"`
- `"too_dark"`
- `"low_contrast"`

---

### 3) Local LLM Reasoning Layer  

- Model: **`google/flan-t5-small`** (fully local, CPU-friendly)  
- Input: structured features only  
- Tasks:
  1. Judge suitability for professional e-commerce use  
  2. List concrete issues  
  3. Provide a concise reasoning summary  
  4. Output a binary verdict (`Suitable` / `Not suitable`)  
  5. Estimate confidence in [0,1]

The response is constrained to **valid JSON**, ensuring downstream usability.

---

### 4) Inference Orchestration  

The `InferencePipeline` class coordinates:

1. Image loading  
2. Parallel feature extraction  
3. Structured signal assembly  
4. LLM reasoning  
5. Result aggregation  

Each image produces one independent JSON file in `data/outputs/`.

---

## Sample Output Schema  

```json
{
  "image_quality_score": 0.74,
  "issues_detected": ["blurred", "too_dark"],
  "detected_objects": ["shoe", "hand"],
  "text_detected": [],
  "llm_reasoning_summary": "The image appears informal with poor lighting...",
  "final_verdict": "Not suitable",
  "confidence": 0.82
}

Models Used
| Task       | Model         | Justification                            |
| ---------- | ------------- | ---------------------------------------- |
| Detection  | YOLOv8n       | Fast, lightweight, strong generalization |
| Embeddings | CLIP ViT-B/32 | Standard multimodal representation       |
| OCR        | Tesseract     | Reliable baseline for text extraction    |
| LLM        | flan-t5-small | Practical for local inference on CPU     |


Running The Sysytem

python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt


Engineering Trade-offs

| Choice                         | Rationale                                    |
| ------------------------------ | -------------------------------------------- |
| YOLOv8n over larger detectors  | Prioritized speed and local usability        |
| CLIP embeddings retained       | Enables future retrieval or similarity tasks |
| flan-t5-small over larger LLMs | Feasible local execution without GPUs        |
| Structured JSON output         | Reliability over free-form text              |


Limitations
1.Local LLM reasoning depth

  flan-t5-small is weaker than GPT-class models
  Future work: upgrade to flan-t5-base or add schema validation + retries

2.OCR robustness

 Struggles with curved or stylized text
 Future: replace with PaddleOCR or EasyOCR

3.Background understanding

 No explicit segmentation
 Future: integrate SAM or DeepLab for clutter detection

4.Throughput

 Currently single-image processing
 Future: batching + GPU acceleration


Author

Aman Shukla
Machine Learning Engineer — Computer Vision & Generative AI