# Blueprint-AI-Estimator

# An AI-powered Quantity Takeoff (QTO) copilot that automates door, window, and zone extraction from construction blueprints using YOLOv8 and Streamlit.

#  ARK Takeoff Copilot (AutoQTO-Vision)

An end-to-end AI application designed to automate the manual Quantity Takeoff (QTO) process for construction estimators. 

By leveraging a custom-trained YOLOv8 computer vision model, this tool instantly processes architectural floor plans, detects structural features, and generates structured, downloadable estimation reports.

## The Business Problem
Currently, construction estimators spend hours manually parsing complex, often messy 2D blueprints—clicking, counting, and entering data into Excel to quantify doors, windows, and room areas. 

**The Solution:** This AI Copilot reduces a 45-minute manual counting task to roughly 150 milliseconds. It provides a human-in-the-loop validation interface and automates the data pipeline directly to `.xlsx`, eliminating human counting errors and accelerating the estimation workflow.

## Core Features
* **Automated Feature Extraction:** Custom YOLOv8 Nano model trained to detect `doors`, `windows`, and `zones` on 2D CAD layouts and PDFs.
* **Human-in-the-Loop Validation:** An interactive Streamlit dashboard that overlays bounding boxes on the original blueprint, allowing estimators to visually verify AI accuracy.
* **Instant QTO Generation:** A Pandas-driven backend that calculates exact counts and dynamically generates a formatted Excel (`.xlsx`) takeoff report for immediate download.
* **Edge-Optimized:** Configured for local, CPU-bound execution (tested on AMD Ryzen 5) to handle sensitive client blueprints without relying on external cloud APIs.

## Technology Stack
* **Computer Vision:** Ultralytics YOLOv8, OpenCV, PyTorch (CPU-Optimized)
* **Data & Analytics:** Pandas, OpenPyXL
* **Frontend UI:** Streamlit
* **Dataset:** 402 Architectural Blueprint Images | Dataset link: https://universe.roboflow.com/tradeplane/floor-plans-500-jbmh9

## Output Accuracy & Model Performance
The custom YOLOv8 Nano model was trained on 402 architectural blueprint images and validated on an unseen test set. It achieves high precision on critical estimation elements like doors and room boundaries (zones), processing complete layouts in **~130 milliseconds** on a standard CPU.
# Notes Accuracy Depend on Dataset I used this Dataset utilizing CPU of Ryzen 5U processor 

| Detection Class | mAP50 (Accuracy) | Precision | Recall |
| :--- | :--- | :--- | :--- |
| **Doors** | **83.1%** | 82.2% | 81.2% |
| **Zones (Rooms)** | **80.2%** | 70.4% | 81.7% |
| **Windows** | **60.9%** | 71.1% | 59.0% |
| **Overall Model** | **74.7%** | 74.5% | 74.0% |  
