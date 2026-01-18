# Engineering Trade-offs

## 1) Local LLM vs Cloud API

**Chosen:** Local LLM (`flan-t5-small`)

### Advantages
- No API key required  
- Fully offline  
- Reproducible  
- No cost per request  
- Privacy-preserving  

### Disadvantages
- Weaker reasoning than GPT-4  
- Slower on CPU  
- Less flexible formatting  

**Why this choice is reasonable**
- The task allows local models
- Demonstrates ML engineering rather than API dependency
- Avoids external constraints

---

## 2) YOLOv8n vs Larger Detector

**Chosen:** `yolov8n` (nano model)

### Trade-off
- Faster, lighter, less accurate  
vs  
- Slower, heavier, more accurate

**Rationale**
- For this task, high recall is less important than speed
- Running on a normal laptop is a priority
- Object labels are used as context, not final decision makers

---

## 3) CLIP Embeddings vs No Embeddings

**Chosen:** Include CLIP embeddings

### Why include them
- Future extensibility:
  - Image similarity search
  - Deduplication
  - Retrieval
- Shows real multimodal understanding

### Why they are not heavily used now
- The task focuses on explainable reasoning
- Raw embeddings are not human-readable

---

## 4) Hard Rules vs LLM Judgment

We did **not** rely only on rules.

- Quality metrics provide **objective signals**
- LLM provides **subjective reasoning**

This balance:
- Avoids “LLM hallucination”
- Avoids brittle rule systems

---

## 5) Speed vs Explainability

We optimized for:
- Clear signal extraction
- Understandable decision process  

rather than:
- Maximum throughput  
- Maximum accuracy

This is appropriate for an evaluation task.

---

## 6) JSON Strictness vs Flexibility

We enforced structured JSON output.

Trade-off:
- Less creative freedom for LLM  
- More reliability for downstream systems  

This is the correct choice for ML products.
