import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import os

st.set_page_config(layout="wide", page_title="ARK Simplify QTO Extractor")

st.title("AI Quantity Takeoff (QTO) Extractor")
st.markdown("Upload a blueprint to automatically detect **Doors, Windows, and Zones** for estimation.")

# Point to the model you just finished training!
model_path = 'runs/detect/train/weights/best.pt'

uploaded_file = st.file_uploader("Upload Blueprint (Image)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_img.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Blueprint")
        st.image("temp_img.png", use_column_width=True)
        
    with col2:
        st.subheader("AI Detection & Takeoff")
        
        if not os.path.exists(model_path):
            st.warning("⚠️ Model training is not finished yet! Waiting for best.pt...")
        else:
            if st.button("Generate Takeoff", type="primary"):
                with st.spinner("Analyzing architectural elements..."):
                    # Load your newly trained model
                    model = YOLO(model_path)
                    
                    # Run the AI on the image
                    results = model("temp_img.png")
                    
                    # Save and display the image with bounding boxes
                    res_plotted = results[0].plot()
                    cv2.imwrite("annotated.png", res_plotted)
                    st.image("annotated.png", caption="AI Annotated Blueprint", use_column_width=True)
                    
                    # Extract the data into a list
                    items = []
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        conf = float(box.conf[0]) 
                        items.append({"Element": class_name.capitalize(), "Quantity": 1, "Confidence": f"{conf:.2f}"})
                        
                    # Convert to a Pandas DataFrame
                    df = pd.DataFrame(items)
                    
                    if not df.empty:
                        # Group by element type to get total counts
                        summary_df = df.groupby('Element')['Quantity'].sum().reset_index()
                        
                        st.success("Extraction Complete!")
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Export to Excel
                        summary_df.to_excel("takeoff_report.xlsx", index=False)
                        with open("takeoff_report.xlsx", "rb") as f:
                            st.download_button(
                                label="📊 Download Excel Takeoff",
                                data=f,
                                file_name="ARK_Takeoff_Report.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    else:
                        st.warning("No doors, windows, or zones were detected.")
                        