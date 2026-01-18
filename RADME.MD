# Mini Multimodal Intelligence System

A lightweight, offline, **multimodal image reasoning pipeline** that separates perception from reasoning, treats the LLM as a secondary component rather than a decision oracle, and explicitly handles model failure instead of hiding it.

This project was built for an ML engineering assessment that emphasizes **model understanding, system design, and engineering judgment — not prompt writing.**

---

## 🎯 What this system does

Given an input image, the pipeline:

1. **Perception layer (true ML work)**
   - Detects objects using **YOLOv8**
   - Extracts text using **Tesseract OCR**
   - Computes image quality metrics:
     - blur variance  
     - brightness  
     - contrast  
     - composite quality score  

2. **Reasoning layer (local LLM)**
   - Sends **structured features (not raw images)** to a local LLM  
   - The LLM is asked to reason in strict JSON format  
   - Output is validated against a schema  

3. **Reliability layer (engineering judgment)**
   - If the LLM returns malformed text (which often happens), the system:
     - detects the failure,
     - logs it explicitly (`"llm_format_error"`), and  
     - falls back to a **vision-based decision** instead of hallucinating.

4. **Final structured output**
   - Always returns valid JSON with:
     - image_quality_score  
     - detected_objects  
     - text_detected  
     - issues_detected  
     - llm_reasoning_summary  
     - final_verdict  
     - confidence  

---

## 🧠 Design philosophy (why this matters)

This project is **not**:

- a single GPT API call  
- a prompt-only solution  
- a notebook demo  
- or an image → text wrapper  

It is designed as a **real ML system** where:

- Vision models do perception.  
- The LLM only reasons over structured signals.  
- The system assumes the LLM is unreliable.  
- Decisions remain explainable and deterministic.

This reflects how real-world ML systems are built in production.

---

## 🔧 Why a **local LLM** instead of an API?

We deliberately used a **local model (`google/flan-t5-base`)** because:

1. **Offline execution**
   - No internet required  
   - No API keys  
   - Fully reproducible  

2. **Engineering realism**
   - Many companies cannot send images or data to external APIs due to privacy.  
   - Using a local model shows system design, not dependency on a black box.

3. **Failure transparency**
   - Small local models are weak at structured JSON generation.  
   - Instead of hiding this, we **designed around it** with validation + fallback logic.

---

## ⚠️ Why does the LLM “fail” sometimes?

This is intentional and realistic:

- `flan-t5-base` is **not designed for strict JSON generation.**
- It often produces partial or malformed text.
- Rather than pretending the model is perfect, we:
  - detect invalid outputs,  
  - label them as `"llm_format_error"`, and  
  - base the final decision on measurable vision quality.

This demonstrates **uncertainty handling and robustness**, which is exactly what the assessment rewards.

---

## 🏗️ System architecture

