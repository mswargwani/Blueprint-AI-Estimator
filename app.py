import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import io

st.set_page_config(page_title="Takeoff Copilot", layout="wide")
st.title("AI Estimator Copilot")
st.markdown("Automated Quantity Take-off (QTO) and Cost Estimation Workflow.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Project Configuration")
currency = st.sidebar.selectbox("Project Currency", ["INR", "USD", "GBP", "EUR"])

# Auto-scale baseline costs based on currency (assuming base USD)
multiplier = 80 if currency == "INR" else (0.8 if currency == "GBP" else (0.9 if currency == "EUR" else 1))

labor_rate = st.sidebar.number_input(f"Hourly Labor Rate ({currency})", min_value=0.0, value=float(45 * multiplier), step=10.0)

# --- MASTER DATABASE (All 28 Classes) ---
items = [
    "Corner Sofa", "Dinner Table", "Dishwasher", "Door Window", "Double Bed", 
    "Double Door", "Oven Stove", "Refrigerator", "Showcase Table", "Shower", 
    "Single Bed", "Single Door", "Sink", "Sofa 1P", "Sofa 2P", "Sofa 3P", 
    "Sofa_1P", "Sofa_2P", "Sofa_3P", "Stairs", "Study Table", "Television", 
    "Toilet", "Wardrobe", "Wash Basin", "Washbasin Cabinet", "Washing Machine", 
    "Window"
]

base_costs = [
    800, 500, 600, 250, 900, 
    400, 700, 1200, 300, 450, 
    400, 150, 150, 250, 400, 600, 
    250, 400, 600, 2500, 200, 500, 
    250, 600, 120, 300, 700, 
    300
]

install_hours = [
    0.5, 0.5, 2.0, 1.5, 1.0, 
    2.5, 1.5, 0.5, 0.5, 3.0, 
    1.0, 1.5, 1.5, 0.5, 0.5, 0.5, 
    0.5, 0.5, 0.5, 16.0, 0.5, 1.0, 
    2.0, 2.0, 1.5, 1.5, 1.0, 
    1.0
]

default_db = pd.DataFrame({
    "Item": items,
    "Material_Cost": [float(c * multiplier) for c in base_costs],
    "Install_Hours": install_hours
})

# --- MAIN UI TABS ---
# Tabs provide a much better UX for handling large databases vs the main workflow
tab1, tab2 = st.tabs(["Takeoff & Estimate", "Pricing Database"])

with tab2:
    st.subheader("Master Pricing Database")
    st.markdown("Edit the base material costs and installation hours for all 28 AI-recognized classes.")
    # Editable dataframe takes full width in the tab
    pricing_df = st.data_editor(default_db, hide_index=True, use_container_width=True, height=600)

with tab1:
    # Load Model
    @st.cache_resource
    def load_model():
        return YOLO("best.pt")
        
    try:
        model = load_model()
    except Exception as e:
        st.warning("Model 'best.pt' not found in the project directory.")
        model = None

    uploaded_file = st.file_uploader("Upload Architectural Blueprint (PNG, JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file and model:
        image = Image.open(uploaded_file)
        
        if st.button("Generate Take-off & Estimate", type="primary"):
            with st.spinner("Processing blueprint..."):
                
                # 1. AI Inference
                results = model(image)
                annotated_img = results[0].plot()
                
                # 2. Side-by-Side UX Layout
                col_img, col_res = st.columns([1.2, 1])
                
                with col_img:
                    st.image(annotated_img, caption="AI Detected Features", use_container_width=True)
                
                with col_res:
                    st.subheader("Bill of Quantities (BOQ)")
                    
                    # Extract items
                    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
                    
                    if not detected_classes:
                        st.info("No items detected in this blueprint.")
                    else:
                        item_counts = pd.Series(detected_classes).value_counts().reset_index()
                        item_counts.columns = ["Item", "Quantity"]
                        
                        # 3. Estimator Math
                        boq_df = pd.merge(item_counts, pricing_df, on="Item", how="left").fillna(0)
                        
                        boq_df[f"Total_Material_Cost_{currency}"] = boq_df["Quantity"] * boq_df["Material_Cost"]
                        boq_df["Total_Labor_Hrs"] = boq_df["Quantity"] * boq_df["Install_Hours"]
                        boq_df[f"Total_Labor_Cost_{currency}"] = boq_df["Total_Labor_Hrs"] * labor_rate
                        boq_df[f"Final_Line_Total_{currency}"] = boq_df[f"Total_Material_Cost_{currency}"] + boq_df[f"Total_Labor_Cost_{currency}"]
                        
                        # Reorder to exact requested columns
                        final_display_df = boq_df[[
                            "Item", "Quantity", "Material_Cost", 
                            f"Total_Material_Cost_{currency}", "Install_Hours", 
                            "Total_Labor_Hrs", f"Total_Labor_Cost_{currency}", 
                            f"Final_Line_Total_{currency}"
                        ]]
                        
                        st.dataframe(final_display_df, hide_index=True, use_container_width=True)
                        
                        grand_total = final_display_df[f"Final_Line_Total_{currency}"].sum()
                        st.success(f"Grand Total Estimate: {currency} {grand_total:,.2f}")
                        
                        # Export options
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            final_display_df.to_excel(writer, index=False, sheet_name="Estimate")
                        
                        st.download_button(
                            label="Download Estimate (.xlsx)",
                            data=excel_buffer.getvalue(),
                            file_name="Estimate_Report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )