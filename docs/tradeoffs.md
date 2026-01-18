# Limitations & Future Improvements

## 1) Local LLM Limitations

Current model: `flan-t5-small`

Limitations:
- Can sometimes produce vague reasoning  
- May not perfectly follow strict JSON format  
- Lower reasoning depth than GPT-4  

Future improvements:
- Upgrade to `flan-t5-base` or `flan-t5-xl`
- Add JSON schema validation layer
- Add retry logic for malformed responses

---

## 2) OCR Reliability

Limitations:
- Tesseract struggles with:
  - Curved text  
  - Stylized fonts  
  - Low-resolution images  

Improvements:
- Replace with PaddleOCR or EasyOCR  
- Add text region detection before OCR  
- Preprocess images more aggressively  

---

## 3) Image Quality Metrics

Current approach uses:
- Blur variance  
- Brightness  
- Contrast  

Limitations:
- Does not detect:
  - Motion blur vs focus blur  
  - Complex lighting issues  
  - Artistic photography quality  

Future:
- Add:
  - Edge sharpness metrics  
  - Color balance analysis  
  - Background segmentation  

---

## 4) No Explicit Background Analysis

Right now, “background clutter” is inferred indirectly.

Future work:
- Use segmentation model (e.g., SAM or DeepLab)
- Explicitly detect clean vs busy backgrounds

---

## 5) No Face Detection or Pose

Currently not included.

Future:
- Add face detection to flag:
  - Privacy risks  
  - Unprofessional images  
- Add pose estimation to detect:
  - Hands covering product  
  - Unnatural product positioning  

---

## 6) Scalability

Current system:
- Processes images one by one  
- Loads all models once  

Production improvements:
- Batch inference  
- Model caching  
- Parallel processing  
- GPU acceleration  
- REST API wrapper  

---

## 7) Evaluation

We do not currently compute quantitative metrics.

Future additions:
- Manual human labeling  
- Precision/recall of suitability predictions  
- A/B comparison of different LLM prompts  
- Confidence calibration  

---

## 8) Deployment

Not included by design.

If productionized:
- Docker container  
- FastAPI service  
- Model versioning  
- Logging & monitoring  
