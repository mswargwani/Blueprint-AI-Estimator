# Blueprint-AI-Estimator
An enterprise-grade AI pipeline designed to automate Construction Quantity Take-off (QTO) and financial estimation. 

## The Problem
Before a construction company can bid on a project, estimators must perform a **Quantity Take-off (QTO)**. This involves spending hours manually panning across dense architectural blueprints to count every single door, window, toilet, and sink. 
* It is a massive bottleneck.
* It is highly prone to human error (missed items cut directly into profit margins).
* It wastes valuable estimator time that should be spent on strategic vendor pricing.

# AI QTO & Estimator Copilot (v2)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

An enterprise-grade **Human-in-the-Loop (HITL)** AI pipeline designed to automate Construction Quantity Take-off (QTO) and commercial financial estimation. 

Version 3.0 transitions this system from a raw machine learning prototype into a legally auditable B2B software architecture. It solves the "black-box AI" problem by wrapping a high-speed YOLO11 neural network inside a strict, deterministic OpenCV verification engine, ensuring that no AI hallucinations ever make it onto a client's financial invoice.


## The V3.0 Enterprise Solution
ARK Simplify utilizes a **4-Stage Verification Pipeline**:
1. **Adaptive Preprocessing:** Automatically detects thin-line CAD drawings and applies CLAHE contrast enhancement.
2. **AI Vision:** A fine-tuned YOLO11s model detects 25 distinct architectural fixtures using C2PSA Spatial Attention.
3. **OpenCV Heuristic Gates:** A deterministic geometry engine filters out AI hallucinations based on spatial constraints and CAD ink density.
4. **HITL Audit & BOQ:** A persistent Streamlit dashboard forces a human estimator to verify low-confidence detections before generating the final multi-sheet Bill of Quantities (BOQ).

## Core System Architecture & Features

### 1. Neural Inference & Adaptive Preprocessing
* **Model Engine:** Powered by YOLO11 Small (`yolo11s.pt`), trained via Distributed Data Parallel (DDP) on Kaggle Dual T4 GPUs.
* **Ontology Normalization:** Automatically maps messy dataset labels (e.g., `Sofa_1P`, `Sofa 1P`) into a clean, 25-class commercial catalog.
* **Adaptive Contrast Fallback:** If the AI detects zero items on the initial pass (common with thin-line grey CAD drawings), the backend automatically triggers a fallback loop. It applies CLAHE (Contrast Limited Adaptive Histogram Equalization), boosts inference resolution to `1024px`, and re-scans at a lower confidence threshold to guarantee output without crashing.

### 2. OpenCV Geometric Triage Engine
Raw bounding boxes from YOLO are passed through 4 deterministic OpenCV gates to eliminate false positives:
* **Max Area Ratio Gate:** Rejects bounding boxes covering >25% of the blueprint, eliminating full-room AI hallucinations.
* **Aspect Ratio Gate:** Enforces real-world architectural physics (e.g., rejecting a "Door" prediction if its bounding box has a 15:1 width-to-height ratio).
* **CAD Ink Density Gate:** Uses Canny Edge detection to calculate the pixel density inside the bounding box, ensuring the AI isn't predicting fixtures on empty white paper.
* **Cross-Class IoU Deduplication:** Suppresses duplicate overlapping boxes when the AI double-counts nested fixtures (e.g., predicting a "Sink" and a "Wash Basin" on the exact same pixels).

### 3. Human-in-the-Loop (HITL) Workflow
* **Visual Confidence Triage:** Detections are drawn dynamically via OpenCV and color-coded by risk:
  * **Green (>75%):** Auto-approved high-confidence fixtures.
  * **Orange (<75%):** Flagged for mandatory human review.
  * **Red:** Rejected anomalies that failed the OpenCV geometric gates.
* **Immutable Audit Trail:** The `Raw_AI_Count` is mathematically locked in the UI. Estimators can only edit the `Final_Approved_Qty` or inject missing items manually, creating a transparent digital ledger.
* **Estimator Sign-Off:** Excel and CSV financial exports are disabled until the Lead Estimator physically checks a legal certification box verifying the audit.

### 4. Dynamic Commercial BOQ Engine
* **Multi-Currency:** Instant toggling between USD, INR, GBP, and EUR. Baseline material rates auto-scale via localized FX multipliers.
* **Financial Rollup:** Calculates material wastage percentages, installation labor hours, contractor overhead, profit margins, and regional tax/GST parameters.
* **Enterprise Export:** Uses `OpenPyXL` to generate a 4-sheet formatted Excel workbook containing:
  1. Executive Summary
  2. Verified BOQ
  3. CSI Division Cost Rollup
  4. Instance-Level AI Telemetry Log


## Tech Stack & Dependencies
* **Backend Inference Engine:** Python 3.10+, PyTorch, Ultralytics (YOLO11), OpenCV-Python (`cv2`), Pandas, Numpy, Pillow.
* **Frontend UI:** Streamlit (`st.session_state` heavily utilized for persistent audit workspaces and dynamic tabular editing).
* **Cloud Training Hardware:** Google Kaggle (Dual NVIDIA Tesla T4 GPUs) utilizing `device=[0,1]` for DDP acceleration.
* **Dataset:** 2,000+ Floor Plan Renderings (Roboflow), processed to fix class imbalance and label duplication.

## Dataset Link: https://universe.roboflow.com/floor-plan-rendering/floor-plan-ai-object-detection
## Output Sceenshot and Pdf: https://drive.google.com/drive/folders/1Zzc3tQFVsEiYHdwE5gWgxaZfhk4yvsDa?usp=sharing
## New V2 Sceenshot: https://drive.google.com/file/d/1778fKLPqLCzNwMl0x9cY8R9z6ALX2pwt/view?usp=drive_link
