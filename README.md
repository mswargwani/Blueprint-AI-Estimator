# Blueprint-AI-Estimator
An enterprise-grade AI pipeline designed to automate Construction Quantity Take-off (QTO) and financial estimation. 

## The Problem
Before a construction company can bid on a project, estimators must perform a **Quantity Take-off (QTO)**. This involves spending hours manually panning across dense architectural blueprints to count every single door, window, toilet, and sink. 
* It is a massive bottleneck.
* It is highly prone to human error (missed items cut directly into profit margins).
* It wastes valuable estimator time that should be spent on strategic vendor pricing.

## The Solution
**Takeoff Copilot** eliminates manual counting and automates the financial math. 
1. **AI Vision:** A custom-trained YOLOv8 computer vision model scans 2D blueprints and instantly detects 28 distinct architectural classes (Doors, Windows, Plumbing fixtures, etc.).
2. **Estimator Engine:** The Streamlit dashboard merges the AI's object count with an interactive, localized pricing database. 
3. **Automated BOQ:** In seconds, it generates a fully calculated Bill of Quantities (BOQ) with material and labor costs, ready for export to Excel.


## Tech Stack & Tools
* **[Python 3.10+](https://www.python.org/):** Core backend logic and data pipelines.
* **[YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) & [PyTorch](https://pytorch.org/):** Computer vision model for real-time object detection on architectural plans.
* **[Streamlit](https://streamlit.io/):** Rapid development of the interactive, web-based estimator dashboard.
* **[Pandas](https://pandas.pydata.org/):** Data manipulation, tensor-to-dataframe conversion, and complex BOQ math.
* **[OpenPyXL](https://openpyxl.readthedocs.io/):** Automated generation of formatted `.xlsx` enterprise reports.
* **[Google Colab (T4 GPU)](https://colab.research.google.com/):** High-performance cloud computing environment used for 50-epoch model training.
* **[Roboflow](https://roboflow.com/):** Dataset aggregation, annotation, and version control for 2,000+ images.

## Dataset Link: https://universe.roboflow.com/floor-plan-rendering/floor-plan-ai-object-detection
## Output Sceenshot and Pdf: https://drive.google.com/drive/folders/1Zzc3tQFVsEiYHdwE5gWgxaZfhk4yvsDa?usp=sharing
