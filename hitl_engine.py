
import io
import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from ultralytics import YOLO

LABEL_NORMALIZER: Dict[str, str] = {
    "Sofa_1P": "Sofa 1P",
    "Sofa_2P": "Sofa 2P",
    "Sofa_3P": "Sofa 3P",
    "single_door": "Single Door",
    "double_door": "Double Door",
    "door": "Single Door",
    "Door": "Single Door",
    "window": "Window",
    "Window": "Window",
    "toilet": "Toilet",
    "Toilet": "Toilet",
    "sink": "Sink",
    "Sink": "Sink",
    "stair": "Stairs",
    "Stairs": "Stairs",
}

MASTER_CATALOG: List[Dict[str, Any]] = [
    {"Item": "Single Door", "Category": "Carpentry & Doors", "CSI_Division": "08 - Openings", "Unit": "EA", "Base_USD_Material": 180.0, "Unit_Install_Hours": 1.5, "Min_Aspect": 0.15, "Max_Aspect": 6.5},
    {"Item": "Double Door", "Category": "Carpentry & Doors", "CSI_Division": "08 - Openings", "Unit": "EA", "Base_USD_Material": 420.0, "Unit_Install_Hours": 2.5, "Min_Aspect": 0.15, "Max_Aspect": 6.5},
    {"Item": "Door Window", "Category": "Carpentry & Doors", "CSI_Division": "08 - Openings", "Unit": "EA", "Base_USD_Material": 290.0, "Unit_Install_Hours": 2.0, "Min_Aspect": 0.10, "Max_Aspect": 8.0},
    {"Item": "Window", "Category": "Glazing & Openings", "CSI_Division": "08 - Openings", "Unit": "EA", "Base_USD_Material": 310.0, "Unit_Install_Hours": 1.2, "Min_Aspect": 0.03, "Max_Aspect": 25.0},
    {"Item": "Toilet", "Category": "Plumbing & Sanitary", "CSI_Division": "22 - Plumbing", "Unit": "EA", "Base_USD_Material": 260.0, "Unit_Install_Hours": 2.0, "Min_Aspect": 0.25, "Max_Aspect": 4.0},
    {"Item": "Sink", "Category": "Plumbing & Sanitary", "CSI_Division": "22 - Plumbing", "Unit": "EA", "Base_USD_Material": 165.0, "Unit_Install_Hours": 1.5, "Min_Aspect": 0.25, "Max_Aspect": 4.0},
    {"Item": "Wash Basin", "Category": "Plumbing & Sanitary", "CSI_Division": "22 - Plumbing", "Unit": "EA", "Base_USD_Material": 140.0, "Unit_Install_Hours": 1.5, "Min_Aspect": 0.25, "Max_Aspect": 4.0},
    {"Item": "Washbasin Cabinet", "Category": "Plumbing & Sanitary", "CSI_Division": "22 - Plumbing", "Unit": "EA", "Base_USD_Material": 340.0, "Unit_Install_Hours": 2.0, "Min_Aspect": 0.20, "Max_Aspect": 5.0},
    {"Item": "Shower", "Category": "Plumbing & Sanitary", "CSI_Division": "22 - Plumbing", "Unit": "EA", "Base_USD_Material": 480.0, "Unit_Install_Hours": 3.0, "Min_Aspect": 0.25, "Max_Aspect": 4.0},
    {"Item": "Stairs", "Category": "Structural & Civil", "CSI_Division": "03 - Structural", "Unit": "FLT", "Base_USD_Material": 2600.0, "Unit_Install_Hours": 16.0, "Min_Aspect": 0.10, "Max_Aspect": 10.0},
    {"Item": "Oven Stove", "Category": "MEP & Appliances", "CSI_Division": "11 - Equipment", "Unit": "EA", "Base_USD_Material": 720.0, "Unit_Install_Hours": 1.5, "Min_Aspect": 0.25, "Max_Aspect": 4.0},
    {"Item": "Refrigerator", "Category": "MEP & Appliances", "CSI_Division": "11 - Equipment", "Unit": "EA", "Base_USD_Material": 1150.0, "Unit_Install_Hours": 0.8, "Min_Aspect": 0.25, "Max_Aspect": 4.0},
    {"Item": "Dishwasher", "Category": "MEP & Appliances", "CSI_Division": "11 - Equipment", "Unit": "EA", "Base_USD_Material": 620.0, "Unit_Install_Hours": 2.0, "Min_Aspect": 0.25, "Max_Aspect": 4.0},
    {"Item": "Washing Machine", "Category": "MEP & Appliances", "CSI_Division": "11 - Equipment", "Unit": "EA", "Base_USD_Material": 680.0, "Unit_Install_Hours": 1.2, "Min_Aspect": 0.25, "Max_Aspect": 4.0},
    {"Item": "Television", "Category": "MEP & Appliances", "CSI_Division": "11 - Equipment", "Unit": "EA", "Base_USD_Material": 520.0, "Unit_Install_Hours": 1.0, "Min_Aspect": 0.05, "Max_Aspect": 16.0},
    {"Item": "Single Bed", "Category": "FF&E / Interior Fit-Out", "CSI_Division": "12 - Furnishings", "Unit": "EA", "Base_USD_Material": 420.0, "Unit_Install_Hours": 1.0, "Min_Aspect": 0.25, "Max_Aspect": 4.0},
    {"Item": "Double Bed", "Category": "FF&E / Interior Fit-Out", "CSI_Division": "12 - Furnishings", "Unit": "EA", "Base_USD_Material": 890.0, "Unit_Install_Hours": 1.2, "Min_Aspect": 0.25, "Max_Aspect": 4.0},
    {"Item": "Wardrobe", "Category": "FF&E / Interior Fit-Out", "CSI_Division": "12 - Furnishings", "Unit": "EA", "Base_USD_Material": 640.0, "Unit_Install_Hours": 2.2, "Min_Aspect": 0.10, "Max_Aspect": 8.0},
    {"Item": "Sofa 1P", "Category": "FF&E / Interior Fit-Out", "CSI_Division": "12 - Furnishings", "Unit": "EA", "Base_USD_Material": 260.0, "Unit_Install_Hours": 0.5, "Min_Aspect": 0.25, "Max_Aspect": 4.0},
    {"Item": "Sofa 2P", "Category": "FF&E / Interior Fit-Out", "CSI_Division": "12 - Furnishings", "Unit": "EA", "Base_USD_Material": 430.0, "Unit_Install_Hours": 0.6, "Min_Aspect": 0.20, "Max_Aspect": 5.0},
    {"Item": "Sofa 3P", "Category": "FF&E / Interior Fit-Out", "CSI_Division": "12 - Furnishings", "Unit": "EA", "Base_USD_Material": 650.0, "Unit_Install_Hours": 0.8, "Min_Aspect": 0.15, "Max_Aspect": 6.0},
    {"Item": "Corner Sofa", "Category": "FF&E / Interior Fit-Out", "CSI_Division": "12 - Furnishings", "Unit": "EA", "Base_USD_Material": 850.0, "Unit_Install_Hours": 1.0, "Min_Aspect": 0.25, "Max_Aspect": 4.0},
    {"Item": "Dinner Table", "Category": "FF&E / Interior Fit-Out", "CSI_Division": "12 - Furnishings", "Unit": "EA", "Base_USD_Material": 540.0, "Unit_Install_Hours": 0.8, "Min_Aspect": 0.25, "Max_Aspect": 4.0},
    {"Item": "Study Table", "Category": "FF&E / Interior Fit-Out", "CSI_Division": "12 - Furnishings", "Unit": "EA", "Base_USD_Material": 220.0, "Unit_Install_Hours": 0.6, "Min_Aspect": 0.20, "Max_Aspect": 4.5},
    {"Item": "Showcase Table", "Category": "FF&E / Interior Fit-Out", "CSI_Division": "12 - Furnishings", "Unit": "EA", "Base_USD_Material": 310.0, "Unit_Install_Hours": 0.8, "Min_Aspect": 0.15, "Max_Aspect": 6.0},
]

FX_RATES: Dict[str, float] = {
    "INR": 83.50,
    "USD": 1.00,
    "GBP": 0.79,
    "EUR": 0.92,
}


def build_default_pricing_dataframe(currency: str = "INR") -> pd.DataFrame:
    mult = FX_RATES.get(currency, 1.0)
    rows = []
    for entry in MASTER_CATALOG:
        rows.append({
            "Item": entry["Item"],
            "Category": entry["Category"],
            "CSI_Division": entry["CSI_Division"],
            "Unit": entry["Unit"],
            "Unit_Material_Cost": round(entry["Base_USD_Material"] * mult, 2),
            "Unit_Install_Hours": float(entry["Unit_Install_Hours"]),
        })
    return pd.DataFrame(rows)


def get_aspect_bounds(item_name: str) -> Tuple[float, float]:
    for entry in MASTER_CATALOG:
        if entry["Item"] == item_name:
            return entry["Min_Aspect"], entry["Max_Aspect"]
    return 0.05, 20.0


class HITLVerificationEngine:
    def __init__(
        self,
        base_conf: float = 0.20,
        triage_conf: float = 0.65,
        max_area_pct: float = 25.0,
        min_area_pct: float = 0.005,
        min_edge_density: float = 0.002,
        iou_dedup_thresh: float = 0.82,
    ):
        self.base_conf = base_conf
        self.triage_conf = triage_conf
        self.max_area_pct = max_area_pct
        self.min_area_pct = min_area_pct
        self.min_edge_density = min_edge_density
        self.iou_dedup_thresh = iou_dedup_thresh

    @staticmethod
    def preprocess_cad_image(rgb_np: np.ndarray) -> Image.Image:
        """
        Applies CLAHE contrast boosting and sharpens thin CAD lines (like door arcs
        on CubiCasa blueprints) so YOLO can detect faint symbols.
        """
        lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        merged = cv2.merge((cl, a_channel, b_channel))
        enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced_rgb)

    @staticmethod
    def compute_iou(box_a: List[int], box_b: List[int]) -> float:
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        union_area = float(area_a + area_b - intersection_area)
        return intersection_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def compute_cad_ink_density(gray_img: np.ndarray, box: List[int]) -> float:
        x1, y1, x2, y2 = box
        crop = gray_img[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
        if crop.size == 0:
            return 0.0
        edges = cv2.Canny(crop, threshold1=30, threshold2=120)
        non_zero = np.count_nonzero(edges)
        return float(non_zero) / float(crop.size)

    def run_audit_pipeline(
        self, model: YOLO, pil_image: Image.Image
    ) -> Dict[str, Any]:
        rgb_np = np.array(pil_image.convert("RGB"))
        gray_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        img_h, img_w, _ = rgb_np.shape
        total_canvas_area = float(img_h * img_w)

        # Pass 1: Standard high-res YOLO inference (imgsz=1024 prevents crushing thin lines)
        results = model(pil_image, conf=self.base_conf, imgsz=1024, verbose=False)
        raw_boxes = results[0].boxes
        used_fallback = False

        # Pass 2 (Adaptive Fallback): If 0 boxes found, enhance CAD contrast & lower conf to 0.05
        if len(raw_boxes) == 0:
            enhanced_pil = self.preprocess_cad_image(rgb_np)
            results = model(enhanced_pil, conf=0.05, imgsz=1024, augment=True, verbose=False)
            raw_boxes = results[0].boxes
            used_fallback = True

        parsed_candidates = []
        for idx, b in enumerate(raw_boxes):
            cls_id = int(b.cls[0])
            raw_label = model.names.get(cls_id, f"Class_{cls_id}")
            clean_label = LABEL_NORMALIZER.get(raw_label, raw_label)
            conf_score = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            parsed_candidates.append({
                "raw_idx": idx + 1,
                "Item": clean_label,
                "Raw_Label": raw_label,
                "Confidence": conf_score,
                "Box": [x1, y1, x2, y2],
            })

        parsed_candidates.sort(key=lambda d: d["Confidence"], reverse=True)

        accepted_boxes: List[List[int]] = []
        instance_records: List[Dict[str, Any]] = []
        annotated_canvas = rgb_np.copy()

        for cand in parsed_candidates:
            item_name = cand["Item"]
            conf = cand["Confidence"]
            x1, y1, x2, y2 = cand["Box"]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            box_area = float(bw * bh)

            area_pct = (box_area / total_canvas_area) * 100.0
            aspect_ratio = round(float(bw) / float(bh), 2)
            ink_density = self.compute_cad_ink_density(gray_np, [x1, y1, x2, y2])
            min_ar, max_ar = get_aspect_bounds(item_name)

            rejection_reason: Optional[str] = None

            if area_pct > self.max_area_pct:
                rejection_reason = f"Oversized Anomaly ({area_pct:.1f}% > {self.max_area_pct}% plan area)"
            elif area_pct < self.min_area_pct:
                rejection_reason = f"Micro Noise Artifact ({area_pct:.4f}% plan area)"
            elif not (min_ar <= aspect_ratio <= max_ar):
                rejection_reason = f"Invalid Aspect Ratio ({aspect_ratio}:1 outside [{min_ar}-{max_ar}])"
            elif ink_density < self.min_edge_density:
                rejection_reason = f"Blank Canvas Hallucination (Ink density {ink_density:.4f})"
            else:
                for prev_box in accepted_boxes:
                    if self.compute_iou([x1, y1, x2, y2], prev_box) >= self.iou_dedup_thresh:
                        rejection_reason = f"Duplicate Overlapping Box (IoU >= {self.iou_dedup_thresh})"
                        break

            if rejection_reason is not None:
                triage_status = "REJECTED_HEURISTIC"
                color = (220, 53, 69)
                thickness = 1
            elif conf >= self.triage_conf:
                triage_status = "AUTO_APPROVED"
                color = (25, 135, 84)
                thickness = 2
                accepted_boxes.append([x1, y1, x2, y2])
            else:
                triage_status = "FLAGGED_HITL_REVIEW"
                color = (255, 140, 0)
                thickness = 2
                accepted_boxes.append([x1, y1, x2, y2])

            cv2.rectangle(annotated_canvas, (x1, y1), (x2, y2), color, thickness)
            tag_text = f"#{cand['raw_idx']} {item_name} ({int(conf * 100)}%)"
            cv2.putText(
                annotated_canvas,
                tag_text,
                (x1, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

            instance_records.append({
                "Detection_ID": f"DET-{cand['raw_idx']:03d}",
                "Item": item_name,
                "Confidence_Pct": round(conf * 100.0, 1),
                "Triage_Status": triage_status,
                "Heuristic_Gate_Notes": rejection_reason if rejection_reason else "Passed All 4 OpenCV Geometry Gates",
                "Box_Width_Px": bw,
                "Box_Height_Px": bh,
                "Plan_Area_Pct": round(area_pct, 3),
                "Aspect_Ratio": aspect_ratio,
                "CAD_Ink_Density": round(ink_density, 4),
                "BBox_Coords": f"[{x1}, {y1}, {x2}, {y2}]",
            })

        instance_df = pd.DataFrame(instance_records)
        summary_df = self._build_hitl_summary_table(instance_df)

        return {
            "annotated_image": annotated_canvas,
            "instance_df": instance_df,
            "summary_df": summary_df,
            "image_dimensions": (img_w, img_h),
            "used_fallback": used_fallback,
        }

    @staticmethod
    def _build_hitl_summary_table(instance_df: pd.DataFrame) -> pd.DataFrame:
        if instance_df.empty:
            return pd.DataFrame(columns=[
                "Item", "Raw_AI_Detections", "Rejected_By_OpenCV",
                "Auto_Approved_High_Conf", "Flagged_Low_Conf",
                "Mean_Confidence_Pct", "HITL_Audit_Status",
                "Final_Approved_Qty", "Estimator_Audit_Notes"
            ])

        grouped = instance_df.groupby("Item").agg(
            Raw_AI_Detections=("Detection_ID", "count"),
            Rejected_By_OpenCV=("Triage_Status", lambda s: int((s == "REJECTED_HEURISTIC").sum())),
            Auto_Approved_High_Conf=("Triage_Status", lambda s: int((s == "AUTO_APPROVED").sum())),
            Flagged_Low_Conf=("Triage_Status", lambda s: int((s == "FLAGGED_HITL_REVIEW").sum())),
            Mean_Confidence_Pct=("Confidence_Pct", lambda c: round(float(c.mean()), 1)),
        ).reset_index()

        grouped["Final_Approved_Qty"] = grouped["Auto_Approved_High_Conf"] + grouped["Flagged_Low_Conf"]

        def assign_status(row: pd.Series) -> str:
            if row["Flagged_Low_Conf"] > 0:
                return f"REVIEW REQUIRED ({row['Flagged_Low_Conf']} Low-Conf)"
            if row["Rejected_By_OpenCV"] > 0 and row["Final_Approved_Qty"] == 0:
                return "ALL REJECTED BY OPENCV"
            return "VERIFIED (High Confidence)"

        grouped["HITL_Audit_Status"] = grouped.apply(assign_status, axis=1)
        grouped["Estimator_Audit_Notes"] = ""

        grouped.sort_values(
            by=["Flagged_Low_Conf", "Final_Approved_Qty"],
            ascending=[False, False],
            inplace=True,
        )
        return grouped.reset_index(drop=True)

def compute_commercial_boq(
    verified_hitl_df: pd.DataFrame,
    pricing_df: pd.DataFrame,
    currency: str,
    labor_rate: float,
    wastage_pct: float,
    overhead_pct: float,
    profit_margin_pct: float,
    tax_pct: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if verified_hitl_df.empty:
        return pd.DataFrame(), {}

    merged = pd.merge(
        verified_hitl_df[[
            "Item", "Raw_AI_Detections", "Rejected_By_OpenCV",
            "Flagged_Low_Conf", "Final_Approved_Qty",
            "HITL_Audit_Status", "Estimator_Audit_Notes"
        ]],
        pricing_df,
        on="Item",
        how="left",
    ).fillna({
        "Category": "Uncategorized",
        "CSI_Division": "00 - General",
        "Unit": "EA",
        "Unit_Material_Cost": 0.0,
        "Unit_Install_Hours": 0.0,
    })

    merged = merged[merged["Final_Approved_Qty"] > 0].copy()
    if merged.empty:
        return pd.DataFrame(), {}

    wastage_factor = 1.0 + (wastage_pct / 100.0)

    merged["Net_Material_Cost"] = (merged["Final_Approved_Qty"] * merged["Unit_Material_Cost"]).round(2)
    merged["Material_With_Wastage"] = (merged["Net_Material_Cost"] * wastage_factor).round(2)
    merged["Total_Labor_Hours"] = (merged["Final_Approved_Qty"] * merged["Unit_Install_Hours"]).round(2)
    merged["Total_Labor_Cost"] = (merged["Total_Labor_Hours"] * labor_rate).round(2)
    merged["Line_Direct_Total"] = (merged["Material_With_Wastage"] + merged["Total_Labor_Cost"]).round(2)

    total_material = float(merged["Material_With_Wastage"].sum())
    total_labor_cost = float(merged["Total_Labor_Cost"].sum())
    total_labor_hrs = float(merged["Total_Labor_Hours"].sum())
    direct_subtotal = total_material + total_labor_cost

    overhead_amount = round(direct_subtotal * (overhead_pct / 100.0), 2)
    profit_amount = round((direct_subtotal + overhead_amount) * (profit_margin_pct / 100.0), 2)
    pre_tax_bid = direct_subtotal + overhead_amount + profit_amount
    tax_amount = round(pre_tax_bid * (tax_pct / 100.0), 2)
    grand_total_bid = round(pre_tax_bid + tax_amount, 2)

    financial_summary = {
        "currency": currency,
        "total_material_cost": round(total_material, 2),
        "total_labor_hours": round(total_labor_hrs, 2),
        "total_labor_cost": round(total_labor_cost, 2),
        "direct_cost_subtotal": round(direct_subtotal, 2),
        "overhead_amount": overhead_amount,
        "profit_amount": profit_amount,
        "pre_tax_bid": round(pre_tax_bid, 2),
        "tax_amount": tax_amount,
        "grand_total_bid": grand_total_bid,
        "total_verified_units": int(merged["Final_Approved_Qty"].sum()),
    }

    return merged, financial_summary


def generate_enterprise_excel_workbook(
    project_metadata: Dict[str, str],
    financial_summary: Dict[str, Any],
    boq_df: pd.DataFrame,
    instance_df: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    curr = financial_summary.get("currency", "INR")

    exec_rows = [
        {"Section": "Project Information", "Metric": "Project Name", "Value": project_metadata.get("project_name", "N/A")},
        {"Section": "Project Information", "Metric": "Client / Organization", "Value": project_metadata.get("client_name", "N/A")},
        {"Section": "Project Information", "Metric": "Lead Estimator (HITL Auditor)", "Value": project_metadata.get("estimator_name", "N/A")},
        {"Section": "Project Information", "Metric": "Drawing / Blueprint Reference", "Value": project_metadata.get("drawing_ref", "N/A")},
        {"Section": "Project Information", "Metric": "Audit Timestamp (UTC)", "Value": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")},
        {"Section": "Financial Rollup", "Metric": f"Total Material Cost (incl. Wastage) [{curr}]", "Value": financial_summary.get("total_material_cost", 0.0)},
        {"Section": "Financial Rollup", "Metric": "Total Installation Labor Hours [Hrs]", "Value": financial_summary.get("total_labor_hours", 0.0)},
        {"Section": "Financial Rollup", "Metric": f"Total Installation Labor Cost [{curr}]", "Value": financial_summary.get("total_labor_cost", 0.0)},
        {"Section": "Financial Rollup", "Metric": f"Direct Cost Subtotal [{curr}]", "Value": financial_summary.get("direct_cost_subtotal", 0.0)},
        {"Section": "Financial Rollup", "Metric": f"Contractor Overhead & Contingency [{curr}]", "Value": financial_summary.get("overhead_amount", 0.0)},
        {"Section": "Financial Rollup", "Metric": f"Contractor Profit Margin [{curr}]", "Value": financial_summary.get("profit_amount", 0.0)},
        {"Section": "Financial Rollup", "Metric": f"Estimated Tax / GST / VAT [{curr}]", "Value": financial_summary.get("tax_amount", 0.0)},
        {"Section": "Financial Rollup", "Metric": f"GRAND TOTAL COMMERCIAL BID [{curr}]", "Value": financial_summary.get("grand_total_bid", 0.0)},
    ]
    exec_df = pd.DataFrame(exec_rows)

    if not boq_df.empty:
        division_df = boq_df.groupby(["CSI_Division", "Category"]).agg(
            Total_Approved_Units=("Final_Approved_Qty", "sum"),
            Material_Cost=("Material_With_Wastage", "sum"),
            Labor_Hours=("Total_Labor_Hours", "sum"),
            Labor_Cost=("Total_Labor_Cost", "sum"),
            Division_Direct_Total=("Line_Direct_Total", "sum"),
        ).reset_index()
    else:
        division_df = pd.DataFrame()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        exec_df.to_excel(writer, index=False, sheet_name="1_Executive_Summary")
        boq_df.to_excel(writer, index=False, sheet_name="2_Verified_BOQ")
        division_df.to_excel(writer, index=False, sheet_name="3_CSI_Division_Rollup")
        instance_df.to_excel(writer, index=False, sheet_name="4_AI_OpenCV_Audit_Log")

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for col in worksheet.columns:
                max_len = max((len(str(cell.value or "")) for cell in col), default=12)
                col_letter = col[0].column_letter
                worksheet.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 45)

    return output.getvalue()