
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO

from hitl_engine import (
    FX_RATES,
    MASTER_CATALOG,
    HITLVerificationEngine,
    build_default_pricing_dataframe,
    compute_commercial_boq,
    generate_enterprise_excel_workbook,
)
## Styling 

st.set_page_config(
    page_title="AI QTO & HITL Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
    h1, h2, h3, h4 { font-family: 'Segoe UI', Roboto, Helvetica, sans-serif; font-weight: 600; color: #172b4d; }
    .status-banner {
        background-color: #f4f5f7;
        border-left: 5px solid #0052cc;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 16px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 700; color: #0052cc; }
    </style>
""", unsafe_allow_html=True)

st.title("AI Takeoff & Estimation Platform")
st.markdown(
    "Automated Computer Vision Quantity Take-off (QTO), 4-Stage OpenCV Geometric Verification, "
    "and Human-in-the-Loop (HITL) Commercial Auditing."
)
st.divider()


def init_session_state() -> None:
    defaults = {
        "scan_executed": False,
        "raw_pil_image": None,
        "annotated_image": None,
        "instance_df": pd.DataFrame(),
        "hitl_summary_df": pd.DataFrame(),
        "image_dims": (0, 0),
        "active_currency": "INR",
        "pricing_df": build_default_pricing_dataframe("INR"),
        "auditor_signed_off": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


st.sidebar.header("1. Project & Audit Metadata")
project_name = st.sidebar.text_input("Project Title", value="Tower B - Residential Fit-Out")
client_name = st.sidebar.text_input("Client Organization", value="ARK Developments Pvt. Ltd.")
estimator_name = st.sidebar.text_input("Lead Estimator / Auditor", value="Manav Swargwani")
drawing_ref = st.sidebar.text_input("Blueprint Sheet Ref", value="A-104 (Level 2 Floor Plan)")

st.sidebar.divider()
st.sidebar.header("2. Commercial & Financial Rules")

selected_currency = st.sidebar.selectbox("Project Currency", list(FX_RATES.keys()), index=0)

if selected_currency != st.session_state.active_currency:
    st.session_state.active_currency = selected_currency
    st.session_state.pricing_df = build_default_pricing_dataframe(selected_currency)

fx_mult = FX_RATES[selected_currency]
labor_rate = st.sidebar.number_input(
    f"Skilled Labor Rate ({selected_currency}/Hr)",
    min_value=0.0,
    value=float(round(6.0 * fx_mult if selected_currency == "INR" else 45.0 * fx_mult, 2)),
    step=float(round(1.0 * fx_mult, 2)),
)

col_s1, col_s2 = st.sidebar.columns(2)
wastage_pct = col_s1.number_input("Material Wastage (%)", min_value=0.0, max_value=35.0, value=5.0, step=1.0)
overhead_pct = col_s2.number_input("Overhead (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)

col_s3, col_s4 = st.sidebar.columns(2)
profit_pct = col_s3.number_input("Profit Margin (%)", min_value=0.0, max_value=50.0, value=12.0, step=1.0)
tax_pct = col_s4.number_input("Tax / GST (%)", min_value=0.0, max_value=30.0, value=18.0, step=1.0)

st.sidebar.divider()
st.sidebar.header("3. Neural & OpenCV Heuristic Gates")

base_conf = st.sidebar.slider(
    "YOLO Base Detection Sensitivity",
    min_value=0.05, max_value=0.90, value=0.15, step=0.05,
    help="Minimum confidence required for the neural network to propose a candidate box.",
)
triage_conf = st.sidebar.slider(
    "HITL Auto-Approval Gate",
    min_value=0.50, max_value=0.95, value=0.75, step=0.05,
    help="Detections below this threshold are highlighted in Amber and flagged for mandatory human verification.",
)
max_area_pct = st.sidebar.slider(
    "Max Fixture Area Ratio (%)",
    min_value=5.0, max_value=50.0, value=20.0, step=2.5,
    help="OpenCV Rule 1: Automatically rejects any single bounding box covering more than X% of the blueprint.",
)
iou_dedup = st.sidebar.slider(
    "Cross-Class IoU Deduplication",
    min_value=0.50, max_value=0.95, value=0.82, step=0.02,
    help="OpenCV Rule 4: Suppresses duplicate overlapping boxes across similar classes (e.g. Sink vs Wash Basin).",
)


@st.cache_resource
def load_yolo_weights(weights_path: str = "best.pt"):
    return YOLO(weights_path)

try:
    yolo_model = load_yolo_weights("best.pt")
    model_status_ok = True
except Exception as err:
    model_status_ok = False
    st.error(
        f"Critical System Alert: Model weights 'best.pt' not found in working directory. "
        f"Place your trained YOLO weights in the root folder. Details: {err}"
    )


tab_workspace, tab_inspector, tab_database, tab_telemetry = st.tabs([
    "1. QTO & HITL Audit Workspace",
    "2. Visual Box Inspector & Triage Log",
    "3. Master Cost & CSI Database",
    "4. Model Validation Telemetry",
])

with tab_database:
    st.subheader(f"Master Unit Price & Labor Database ({selected_currency})")
    st.markdown(
        "Adjust unit material prices, CSI divisions, and installation labor norms. "
        "Edits automatically recalculate the active commercial Bill of Quantities."
    )
    st.session_state.pricing_df = st.data_editor(
        st.session_state.pricing_df,
        disabled=["Item", "Category", "CSI_Division", "Unit"],
        hide_index=True,
        width="stretch",
        height=680,
        key="master_pricing_editor",
    )

with tab_telemetry:
    st.subheader("Cloud GPU Training & Validation Telemetry")
    st.markdown(
        "Benchmark metrics recorded across unseen validation floor plans using "
        "Mosaic augmentation, CLAHE contrast normalization, and C2PSA Spatial Attention."
    )
    bench_df = pd.DataFrame({
        "Architectural Class": [
            "Global Model Average (25 Classes)", "Toilet", "Washbasin Cabinet",
            "Sink", "Single Door", "Double Door", "Window"
        ],
        "Precision (P)": ["91.2%", "99.0%", "91.8%", "95.9%", "91.2%", "98.5%", "98.9%"],
        "Recall (R)": ["93.3%", "98.8%", "97.7%", "93.8%", "92.2%", "95.8%", "78.7%"],
        "mAP@50": ["94.7%", "98.5%", "98.9%", "98.6%", "95.9%", "95.5%", "85.1%"],
        "mAP@50-95": ["91.8%", "98.5%", "98.3%", "96.5%", "94.8%", "95.5%", "80.0%"],
        "Inference Strategy": [
            "YOLO + 4-Stage OpenCV Heuristics", "Auto-Approve Gate >= 75%",
            "Auto-Approve Gate >= 75%", "IoU Deduplication Active",
            "Aspect Ratio Verified", "Aspect Ratio Verified", "CLAHE Edge Contrast Enhanced"
        ],
    })
    st.dataframe(bench_df, hide_index=True, width="stretch")

with tab_workspace:
    uploaded_blueprint = st.file_uploader(
        "Upload 2D Architectural Floor Plan (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_blueprint and model_status_ok:
        pil_img = Image.open(uploaded_blueprint).convert("RGB")

        col_btn1, col_btn2 = st.columns([1, 4])
        run_scan = col_btn1.button("Execute AI Takeoff & Heuristic Audit", type="primary", width="stretch")
        if col_btn2.button("Reset Workspace"):
            st.session_state.scan_executed = False
            st.session_state.auditor_signed_off = False
            st.rerun()

        if run_scan:
            with st.spinner("Executing YOLO inference and 4-stage OpenCV geometric verification..."):
                engine = HITLVerificationEngine(
                    base_conf=base_conf,
                    triage_conf=triage_conf,
                    max_area_pct=max_area_pct,
                    iou_dedup_thresh=iou_dedup,
                )
                audit_output = engine.run_audit_pipeline(yolo_model, pil_img)

                st.session_state.raw_pil_image = pil_img
                st.session_state.annotated_image = audit_output["annotated_image"]
                st.session_state.instance_df = audit_output["instance_df"]
                st.session_state.hitl_summary_df = audit_output["summary_df"]
                st.session_state.image_dims = audit_output["image_dimensions"]
                st.session_state.scan_executed = True
                st.session_state.auditor_signed_off = False

    if st.session_state.scan_executed:
        inst_df = st.session_state.instance_df
        total_raw = len(inst_df)
        total_auto = int((inst_df["Triage_Status"] == "AUTO_APPROVED").sum()) if not inst_df.empty else 0
        total_flagged = int((inst_df["Triage_Status"] == "FLAGGED_HITL_REVIEW").sum()) if not inst_df.empty else 0
        total_rejected = int((inst_df["Triage_Status"] == "REJECTED_HEURISTIC").sum()) if not inst_df.empty else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Raw Neural Detections", f"{total_raw} Boxes")
        k2.metric("Auto-Approved (High Conf)", f"{total_auto} Boxes")
        k3.metric("Flagged for Human Review", f"{total_flagged} Boxes")
        k4.metric("Rejected by OpenCV Heuristics", f"{total_rejected} Anomalies")

        st.divider()
        left_col, right_col = st.columns([1.15, 1])

        with left_col:
            st.subheader("Annotated Blueprint Triage Map")
            st.caption(
                f"Resolution: {st.session_state.image_dims[0]}x{st.session_state.image_dims[1]} px | "
                "Green = Auto-Approved | Orange = Flagged for Estimator Review | Red = Rejected Geometry Anomaly"
            )
            st.image(st.session_state.annotated_image, width="stretch")

            with st.expander("Add Unlisted Architectural Fixture to Audit Table"):
                existing_items = st.session_state.hitl_summary_df["Item"].tolist() if not st.session_state.hitl_summary_df.empty else []
                missing_catalog_items = [entry["Item"] for entry in MASTER_CATALOG if entry["Item"] not in existing_items]
                if missing_catalog_items:
                    add_col1, add_col2, add_col3 = st.columns([2, 1, 1])
                    new_item_name = add_col1.selectbox("Select Fixture Class", missing_catalog_items)
                    new_item_qty = add_col2.number_input("Manual Quantity", min_value=1, max_value=500, value=1, step=1)
                    if add_col3.button("Inject Fixture"):
                        new_row = pd.DataFrame([{
                            "Item": new_item_name,
                            "Raw_AI_Detections": 0,
                            "Rejected_By_OpenCV": 0,
                            "Auto_Approved_High_Conf": 0,
                            "Flagged_Low_Conf": 0,
                            "Mean_Confidence_Pct": 100.0,
                            "HITL_Audit_Status": "MANUAL ESTIMATOR INJECTION",
                            "Final_Approved_Qty": int(new_item_qty),
                            "Estimator_Audit_Notes": "Added manually by estimator during HITL review",
                        }])
                        st.session_state.hitl_summary_df = pd.concat(
                            [st.session_state.hitl_summary_df, new_row], ignore_index=True
                        )
                        st.rerun()
                else:
                    st.info("All 25 catalog classes are already present in the current table.")

        with right_col:
            st.subheader("Step 1: Human-in-the-Loop (HITL) Quantity Verification")
            st.markdown(
                "Inspect rows marked **REVIEW REQUIRED**. Double-click cells in **`Final_Approved_Qty`** "
                "or **`Estimator_Audit_Notes`** to override AI predictions. Raw AI counts remain locked for audit compliance."
            )

            edited_hitl_df = st.data_editor(
                st.session_state.hitl_summary_df,
                disabled=[
                    "Item", "Raw_AI_Detections", "Rejected_By_OpenCV",
                    "Auto_Approved_High_Conf", "Flagged_Low_Conf",
                    "Mean_Confidence_Pct", "HITL_Audit_Status"
                ],
                hide_index=True,
                width="stretch",
                key="hitl_table_editor",
            )

            st.session_state.hitl_summary_df = edited_hitl_df

            st.session_state.auditor_signed_off = st.checkbox(
                f"I ({estimator_name}) certify that all flagged low-confidence detections have been audited against sheet {drawing_ref}.",
                value=st.session_state.auditor_signed_off,
            )

        st.divider()
        st.subheader("Step 2: Commercial Bill of Quantities (BOQ) & Bid Rollup")

        boq_df, fin_summary = compute_commercial_boq(
            verified_hitl_df=edited_hitl_df,
            pricing_df=st.session_state.pricing_df,
            currency=selected_currency,
            labor_rate=labor_rate,
            wastage_pct=wastage_pct,
            overhead_pct=overhead_pct,
            profit_margin_pct=profit_pct,
            tax_pct=tax_pct,
        )

        if boq_df.empty:
            st.warning("No verified fixtures with a positive quantity (>0) are currently present in the estimate.")
        else:
            st.dataframe(boq_df, hide_index=True, width="stretch")

            f1, f2, f3, f4, f5 = st.columns(5)
            f1.metric("Material (incl. Wastage)", f"{selected_currency} {fin_summary.get('total_material_cost', 0):,.2f}")
            f2.metric(f"Labor ({fin_summary.get('total_labor_hours', 0):,.1f} Hrs)", f"{selected_currency} {fin_summary.get('total_labor_cost', 0):,.2f}")
            f3.metric("Direct Subtotal", f"{selected_currency} {fin_summary.get('direct_cost_subtotal', 0):,.2f}")
            f4.metric("Overhead + Profit", f"{selected_currency} {(fin_summary.get('overhead_amount', 0) + fin_summary.get('profit_amount', 0)):,.2f}")
            f5.metric("Grand Total Bid (incl. Tax)", f"{selected_currency} {fin_summary.get('grand_total_bid', 0):,.2f}")

            st.divider()
            exp1, exp2 = st.columns(2)

            project_meta = {
                "project_name": project_name,
                "client_name": client_name,
                "estimator_name": estimator_name,
                "drawing_ref": drawing_ref,
            }

            excel_bytes = generate_enterprise_excel_workbook(
                project_metadata=project_meta,
                financial_summary=fin_summary,
                boq_df=boq_df,
                instance_df=st.session_state.instance_df,
            )

            exp1.download_button(
                label="Download 4-Sheet Enterprise Audit Workbook (.xlsx)",
                data=excel_bytes,
                file_name=f"ARK_Verified_BOQ_{project_name.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                disabled=not st.session_state.auditor_signed_off,
                help="Sign off on the HITL audit checkbox above to unlock official commercial exports.",
            )

            csv_bytes = boq_df.to_csv(index=False).encode("utf-8")
            exp2.download_button(
                label="Export Flat BOQ for ERP / Google Sheets (.csv)",
                data=csv_bytes,
                file_name=f"ARK_Verified_BOQ_{project_name.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=not st.session_state.auditor_signed_off,
            )

            if not st.session_state.auditor_signed_off:
                st.caption("Note: Check the Lead Estimator certification box in Step 1 to unlock the official Excel and ERP exports.")

with tab_inspector:
    st.subheader("Instance-Level Detection Telemetry & Visual Crop Inspector")
    if not st.session_state.scan_executed or st.session_state.instance_df.empty:
        st.info("Run an AI Takeoff scan in Tab 1 to inspect individual bounding box detections and OpenCV heuristic telemetry.")
    else:
        inst_df = st.session_state.instance_df

        filter_status = st.multiselect(
            "Filter by Triage Status",
            options=["AUTO_APPROVED", "FLAGGED_HITL_REVIEW", "REJECTED_HEURISTIC"],
            default=["FLAGGED_HITL_REVIEW", "REJECTED_HEURISTIC", "AUTO_APPROVED"],
        )
        filtered_inst = inst_df[inst_df["Triage_Status"].isin(filter_status)]
        st.dataframe(filtered_inst, hide_index=True, width="stretch")

        st.divider()
        st.subheader("Zoomed CAD Symbol Crop Verification")
        selected_det_id = st.selectbox(
            "Select a Detection ID to inspect its high-resolution blueprint crop:",
            options=filtered_inst["Detection_ID"].tolist(),
        )
        if selected_det_id and st.session_state.raw_pil_image is not None:
            row_data = filtered_inst[filtered_inst["Detection_ID"] == selected_det_id].iloc[0]
            coords = [int(x.strip()) for x in row_data["BBox_Coords"].strip("[]").split(",")]
            x1, y1, x2, y2 = coords
            raw_np = np.array(st.session_state.raw_pil_image)
            h, w, _ = raw_np.shape
            pad = 35
            crop_np = raw_np[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]

            c_crop, c_meta = st.columns([1, 2])
            with c_crop:
                st.image(crop_np, caption=f"{selected_det_id}: {row_data['Item']} (Padded 35px)", width=280)
            with c_meta:
                st.markdown(f"""
                * **Detected Class:** `{row_data['Item']}`
                * **Confidence Score:** `{row_data['Confidence_Pct']}%`
                * **Triage Gate Status:** `{row_data['Triage_Status']}`
                * **OpenCV Heuristic Gate Notes:** `{row_data['Heuristic_Gate_Notes']}`
                * **Bounding Box Dimensions:** `{row_data['Box_Width_Px']}px × {row_data['Box_Height_Px']}px` (`{row_data['Plan_Area_Pct']}%` of canvas)
                * **Aspect Ratio & CAD Ink Density:** `{row_data['Aspect_Ratio']}:1` | Ink Density: `{row_data['CAD_Ink_Density']}`
                """)