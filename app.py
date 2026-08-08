"""
app.py
------
Flask web application for GUB Academic AI.
Serves five pages: Dashboard, Predict, Result, Model Performance, About.

Run:  python app.py
"""

import io
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, send_file

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "gub-academic-ai-secret-2025")

# Expose Python builtins needed in templates
app.jinja_env.globals["enumerate"] = enumerate

# ─── Load saved artefacts ─────────────────────────────────────────────────────
def load_artifacts():
    """Load trained model, scaler, encoders, and metadata from disk."""
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        return None, None, None, None, None

    with open(meta_path) as f:
        metadata = json.load(f)

    model          = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler         = joblib.load(MODELS_DIR / "scaler.pkl")
    encoders       = joblib.load(MODELS_DIR / "encoders.pkl")
    target_encoder = joblib.load(MODELS_DIR / "target_encoder.pkl")

    return model, scaler, encoders, target_encoder, metadata

model, scaler, encoders, target_encoder, metadata = load_artifacts()

# ─── Helpers ──────────────────────────────────────────────────────────────────
DEPARTMENTS  = ["CSE", "EEE", "BBA", "LAW", "English",
                "Economics", "Architecture", "Pharmacy", "Civil", "Textile"]
COURSE_TYPES = ["Theory", "Lab", "Theory+Lab", "Seminar"]
SEMESTERS    = ["Spring", "Summer", "Fall"]

def build_recommendation(demand: str, form: dict) -> str:
    """
    Rule-based planning recommendation derived from the ML prediction
    and selected input features.  NOT a separate AI model.
    """
    enrollment = int(form.get("previous_enrollment", 60))
    sections   = int(form.get("previous_sections", 2))
    lab_req    = int(form.get("lab_required", 0))
    growth     = float(form.get("registration_growth_rate", 5))
    popularity = int(form.get("course_popularity", 5))

    lines = []
    if demand == "HIGH":
        lines.append("⚠️ High resource demand is predicted for this course.")
        rec_sections = max(sections + 1, int(np.ceil(enrollment * (1 + growth/100) / 40)))
        lines.append(f"Consider opening at least {rec_sections} section(s) next semester.")
        if lab_req:
            lines.append("Ensure lab slots are booked well in advance.")
        if growth > 15:
            lines.append("Registration growth is notable — reserve additional classroom capacity.")
    elif demand == "MEDIUM":
        lines.append("ℹ️ Moderate resource demand is predicted.")
        lines.append("Maintain the current number of sections and monitor enrollment closely.")
        if growth > 10:
            lines.append("Consider scheduling one extra section as a buffer.")
        if lab_req:
            lines.append("Confirm lab availability early in the semester.")
    else:  # LOW
        lines.append("✅ Low resource demand is predicted for this course.")
        lines.append("Current resources appear sufficient for the upcoming semester.")
        if popularity >= 7:
            lines.append("Despite low predicted demand, the course is popular — keep an eye on late registrations.")

    return "  ".join(lines)

def estimate_enrollment(form: dict, demand: str) -> int:
    base = int(form.get("previous_enrollment", 60))
    growth = float(form.get("registration_growth_rate", 5))
    factor = {"HIGH": 1.20, "MEDIUM": 1.05, "LOW": 0.92}.get(demand, 1.0)
    return max(10, int(base * (1 + growth / 100) * factor))

def recommend_sections(est_enrollment: int) -> int:
    return max(1, int(np.ceil(est_enrollment / 40)))

def recommend_room(demand: str, est_enrollment: int, sections: int) -> str:
    per_section = int(np.ceil(est_enrollment / max(sections, 1)))
    cap = max(30, per_section + 10)
    ranges = {(0, 40): "30–40", (40, 60): "40–60", (60, 80): "60–80",
              (80, 120): "80–120", (120, 9999): "120+"}
    for (lo, hi), label in ranges.items():
        if lo <= cap < hi:
            return label
    return "120+"

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Dashboard — summary statistics and interactive analytics."""
    if metadata is None:
        return render_template("index.html", ready=False)

    dc = metadata.get("demand_counts", {})
    total = sum(dc.values())
    all_res = metadata.get("all_model_results", {})
    sorted_models = sorted(all_res.items(), key=lambda x: x[1]["f1_score"], reverse=True)

    # Department demand & popularity distribution for dynamic Chart.js charts
    dept_demand = {}
    pop_dist = {}
    data_path = BASE_DIR / "data" / "university_academic_resource_dataset.csv"
    if data_path.exists():
        try:
            df_data = pd.read_csv(data_path)
            dept_grp = df_data.groupby(["department", "target_resource_demand"]).size().unstack(fill_value=0)
            dept_demand = {dept: {col: int(val) for col, val in row.items()} for dept, row in dept_grp.iterrows()}
            pop_counts = df_data["course_popularity"].value_counts().sort_index()
            pop_dist = {int(k): int(v) for k, v in pop_counts.items()}
        except Exception:
            pass

    ctx = {
        "ready":          True,
        "total_courses":  total,
        "total_records":  metadata["dataset_shape"][0],
        "high_demand":    dc.get("HIGH", 0),
        "medium_demand":  dc.get("MEDIUM", 0),
        "low_demand":     dc.get("LOW", 0),
        "best_model":     metadata["best_model_name"],
        "best_f1":        f"{metadata['best_model_metrics']['f1_score']:.4f}",
        "best_accuracy":  f"{metadata['best_model_metrics']['accuracy']:.4f}",
        "best_precision": f"{metadata['best_model_metrics']['precision']:.4f}",
        "best_recall":    f"{metadata['best_model_metrics']['recall']:.4f}",
        "all_results":    all_res,
        "sorted_models":  sorted_models,
        "dept_demand":    dept_demand,
        "pop_dist":       pop_dist,
    }
    return render_template("index.html", **ctx)


@app.route("/predict", methods=["GET"])
def predict():
    """Prediction form."""
    return render_template("predict.html",
                           departments=DEPARTMENTS,
                           course_types=COURSE_TYPES,
                           semesters=SEMESTERS)


@app.route("/predict", methods=["POST"])
def predict_post():
    """Handle form submission → run model → redirect to result."""
    if model is None:
        return render_template("predict.html",
                               error="Model not trained yet. Please run train_model.py first.",
                               departments=DEPARTMENTS,
                               course_types=COURSE_TYPES,
                               semesters=SEMESTERS)

    form = request.form

    # ── Encode categorical inputs ──────────────────────────────────────────────
    try:
        dept_enc  = encoders["department"].transform([form["department"]])[0]
        ctype_enc = encoders["course_type"].transform([form["course_type"]])[0]
        sem_enc   = encoders["semester"].transform([form["semester"]])[0]
    except ValueError:
        dept_enc  = 0
        ctype_enc = 0
        sem_enc   = 0

    # ── Build feature vector (same order & columns as training) ────────────────
    feature_dict = {
        "department":              dept_enc,
        "course_type":             ctype_enc,
        "credit":                  int(form.get("credit", 3)),
        "semester":                sem_enc,
        "previous_enrollment":     int(form.get("previous_enrollment", 60)),
        "previous_sections":       int(form.get("previous_sections", 2)),
        "previous_pass_rate":      float(form.get("previous_pass_rate", 75)),
        "course_popularity":       int(form.get("course_popularity", 5)),
        "lab_required":            int(form.get("lab_required", 0)),
        "lab_capacity":            int(form.get("lab_capacity", 0)),
        "room_capacity":           int(form.get("room_capacity", 50)),
        "instructor_count":        int(form.get("instructor_count", 2)),
        "class_duration":          int(form.get("class_duration", 90)),
        "previous_resource_usage": float(form.get("previous_resource_usage", 60)),
        "registration_growth_rate":float(form.get("registration_growth_rate", 5)),
    }
    features_df = pd.DataFrame([feature_dict])[metadata["features"]]

    features_scaled = scaler.transform(features_df)
    pred_enc = model.predict(features_scaled)[0]
    demand   = target_encoder.inverse_transform([pred_enc])[0]

    # ── Probabilities (if available) ──────────────────────────────────────────
    try:
        proba = model.predict_proba(features_scaled)[0]
        label_proba = {
            lbl: round(float(p) * 100, 1)
            for lbl, p in zip(target_encoder.classes_, proba)
        }
    except Exception:
        label_proba = {demand: 100.0}

    # ── Planning estimates ────────────────────────────────────────────────────
    est_enr  = estimate_enrollment(form, demand)
    rec_sec  = recommend_sections(est_enr)
    rec_room = recommend_room(demand, est_enr, rec_sec)
    rec_lab  = "Required" if int(form.get("lab_required", 0)) else "Not Required"
    rec_note = build_recommendation(demand, form)

    # ── Store in session and redirect ─────────────────────────────────────────
    session["result"] = {
        "demand":           demand,
        "label_proba":      label_proba,
        "est_enrollment":   est_enr,
        "rec_sections":     rec_sec,
        "rec_room":         rec_room,
        "rec_lab":          rec_lab,
        "rec_note":         rec_note,
        "planning_priority":demand,
        "form": {
            "department":             form.get("department"),
            "course_code":            form.get("course_code"),
            "course_type":            form.get("course_type"),
            "credit":                 form.get("credit"),
            "semester":               form.get("semester"),
            "previous_enrollment":    form.get("previous_enrollment"),
            "previous_sections":      form.get("previous_sections"),
            "previous_pass_rate":     form.get("previous_pass_rate"),
            "course_popularity":      form.get("course_popularity"),
            "lab_required":           form.get("lab_required"),
            "lab_capacity":           form.get("lab_capacity", "0"),
            "room_capacity":          form.get("room_capacity", "50"),
            "instructor_count":       form.get("instructor_count", "2"),
            "class_duration":         form.get("class_duration", "90"),
            "previous_resource_usage": form.get("previous_resource_usage", "60"),
            "registration_growth_rate": form.get("registration_growth_rate"),
        },
    }
    return redirect(url_for("result"))


def generate_pdf_report(data: dict, meta: dict) -> bytes:
    """
    Generate a full PDF prediction report using ReportLab Platypus.
    Includes headers, hero prediction badge, probabilities, planning estimates,
    rule-based recommendations, 16 complete submitted input parameters,
    production model evaluation metrics, and disclaimer.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155')
    )

    heading_style = ParagraphStyle(
        'SectionHeadingCustom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=6,
        spaceAfter=3
    )

    elements = []

    # 1. Header Table
    timestamp = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    header_html = (
        "<b><font size=14 color='#0F172A'>GUB ACADEMIC AI</font></b><br/>"
        "<b><font size=9.5 color='#2563EB'>Academic Resource Demand Prediction & Planning System</font></b><br/>"
        "<font size=8 color='#64748B'>Green University of Bangladesh • Institutional Analytics Prototype</font>"
    )
    time_html = f"<b><font size=8.5 color='#0F172A'>Report Generated:</font></b><br/><font size=8 color='#64748B'>{timestamp}</font>"
    
    header_table = Table([[Paragraph(header_html, styles['Normal']), Paragraph(time_html, ParagraphStyle('TimeRight', parent=styles['Normal'], alignment=2))]], colWidths=[370, 170])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4)
    ]))
    elements.append(header_table)
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=10))

    # 2. Prediction Hero Result Box
    demand = data.get("demand", "UNKNOWN")
    label_proba = data.get("label_proba", {})
    confidence = label_proba.get(demand, 100.0)

    if demand == "HIGH":
        badge_bg = colors.HexColor('#FEF2F2')
        badge_border = colors.HexColor('#EF4444')
        badge_text_color = '#DC2626'
    elif demand == "MEDIUM":
        badge_bg = colors.HexColor('#FFFBEB')
        badge_border = colors.HexColor('#F59E0B')
        badge_text_color = '#D97706'
    else:
        badge_bg = colors.HexColor('#ECFDF5')
        badge_border = colors.HexColor('#10B981')
        badge_text_color = '#059669'

    res_html = (
        f"<font size=8.5 color='#64748B'><b>AI RESOURCE DEMAND PREDICTION</b></font><br/><br/>"
        f"<font size=18 color='{badge_text_color}'><b>{demand} DEMAND</b></font><br/><br/>"
        f"<font size=9 color='#334155'>Model Confidence: <b>{confidence:.1f}%</b></font>"
    )

    proba_rows = [["Demand Level", "Probability"]]
    for k, v in label_proba.items():
        proba_rows.append([k, f"{v:.1f}%"])
    
    proba_table = Table(proba_rows, colWidths=[90, 60])
    proba_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))

    result_box_table = Table([[Paragraph(res_html, styles['Normal']), proba_table]], colWidths=[360, 180])
    result_box_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), badge_bg),
        ('BOX', (0,0), (-1,-1), 1, badge_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(result_box_table)
    elements.append(Spacer(1, 8))

    # 3. Estimated Planning Requirements Table
    elements.append(Paragraph("Estimated Planning Requirements", heading_style))
    plan_data = [
        ["Expected Enrollment", "Recommended Sections", "Room Capacity", "Lab Facility", "Planning Priority"],
        [
            str(data.get("est_enrollment", "N/A")),
            str(data.get("rec_sections", "N/A")),
            str(data.get("rec_room", "N/A")),
            str(data.get("rec_lab", "N/A")),
            str(data.get("planning_priority", "N/A"))
        ]
    ]
    plan_table = Table(plan_data, colWidths=[108, 108, 108, 108, 108])
    plan_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 7.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#F8FAFC')),
        ('FONTNAME', (0,1), (-1,1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,1), (-1,1), 9.5),
        ('TEXTCOLOR', (0,1), (-1,1), colors.HexColor('#1E293B')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(plan_table)
    elements.append(Spacer(1, 8))

    # 4. Rule-Based Planning Recommendation
    elements.append(Paragraph("Planning Recommendation (Rule-Based Optimization System)", heading_style))
    rec_text = Paragraph(data.get("rec_note", "No recommendation generated."), body_style)
    rec_table = Table([[rec_text]], colWidths=[540])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F0FDF4')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#86EFAC')),
        ('PADDING', (0,0), (-1,-1), 7),
    ]))
    elements.append(rec_table)
    elements.append(Spacer(1, 8))

    # 5. Submitted Input Parameters (16 fields complete profile)
    elements.append(Paragraph("Submitted Input Parameters (16 Profile Fields)", heading_style))
    f = data.get("form", {})
    input_rows = [
        [Paragraph("<b>Department:</b>", body_style), str(f.get("department", "N/A")), Paragraph("<b>Course Popularity (1-10):</b>", body_style), str(f.get("course_popularity", "N/A"))],
        [Paragraph("<b>Course Code:</b>", body_style), str(f.get("course_code", "N/A")), Paragraph("<b>Lab Facility Required:</b>", body_style), "Yes" if str(f.get("lab_required")) == "1" else "No"],
        [Paragraph("<b>Course Type:</b>", body_style), str(f.get("course_type", "N/A")), Paragraph("<b>Lab Seating Capacity:</b>", body_style), str(f.get("lab_capacity", "N/A"))],
        [Paragraph("<b>Credit Hours:</b>", body_style), str(f.get("credit", "N/A")), Paragraph("<b>Classroom Capacity:</b>", body_style), str(f.get("room_capacity", "N/A"))],
        [Paragraph("<b>Target Semester:</b>", body_style), str(f.get("semester", "N/A")), Paragraph("<b>Instructor Count:</b>", body_style), str(f.get("instructor_count", "N/A"))],
        [Paragraph("<b>Previous Enrollment:</b>", body_style), str(f.get("previous_enrollment", "N/A")), Paragraph("<b>Class Session Duration:</b>", body_style), f"{f.get('class_duration', 'N/A')} min"],
        [Paragraph("<b>Previous Sections:</b>", body_style), str(f.get("previous_sections", "N/A")), Paragraph("<b>Prev. Resource Usage Index:</b>", body_style), str(f.get("previous_resource_usage", "N/A"))],
        [Paragraph("<b>Previous Pass Rate (%):</b>", body_style), f"{f.get('previous_pass_rate', 'N/A')}%", Paragraph("<b>Registration Growth Rate (%):</b>", body_style), f"{f.get('registration_growth_rate', 'N/A')}%"],
    ]

    input_table = Table(input_rows, colWidths=[140, 130, 140, 130])
    input_table.setStyle(TableStyle([
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F8FAFC')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#F8FAFC')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    elements.append(input_table)
    elements.append(Spacer(1, 8))

    # 6. Model Evaluation Information
    elements.append(Paragraph("Machine Learning Production Model Information", heading_style))
    best_model_name = meta.get("best_model_name", "Logistic Regression") if meta else "Logistic Regression"
    metrics = meta.get("best_model_metrics", {}) if meta else {}
    
    acc_val = metrics.get('accuracy', 0.8660)
    prec_val = metrics.get('precision', 0.8636)
    rec_val = metrics.get('recall', 0.8660)
    f1_val = metrics.get('f1_score', 0.8638)

    model_data = [
        ["Selected Model", "Accuracy", "Precision", "Recall", "Weighted F1-Score"],
        [
            best_model_name,
            f"{acc_val * 100:.2f}%",
            f"{prec_val * 100:.2f}%",
            f"{rec_val * 100:.2f}%",
            f"{f1_val * 100:.2f}%"
        ]
    ]
    model_table = Table(model_data, colWidths=[140, 100, 100, 100, 100])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(model_table)
    elements.append(Spacer(1, 8))

    # 7. Disclaimer
    disc_text = Paragraph(
        "<b>Synthetic Data Disclaimer:</b> This prototype uses synthetic/representative academic data and does not contain official or confidential GUB institutional data. All prediction outputs and planning estimates are for institutional planning demonstration purposes.",
        ParagraphStyle('Disc', parent=body_style, fontSize=7, leading=9, textColor=colors.HexColor('#64748B'))
    )
    elements.append(disc_text)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


@app.route("/export-report")
def export_report():
    """Generate and return downloadable full PDF prediction report."""
    data = session.get("result")
    if not data:
        return redirect(url_for("predict"))

    pdf_bytes = generate_pdf_report(data, metadata or {})
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="GUB_Academic_AI_Prediction_Report.pdf"
    )



@app.route("/result")
def result():
    """Prediction result page."""
    data = session.get("result")
    if not data:
        return redirect(url_for("predict"))
    return render_template("result.html", **data)


@app.route("/model-performance")
def model_performance():
    """Model comparison and metrics page."""
    if metadata is None:
        return render_template("model_performance.html", ready=False)
    all_res = metadata.get("all_model_results", {})
    sorted_models = sorted(all_res.items(), key=lambda x: x[1]["f1_score"], reverse=True)

    return render_template("model_performance.html",
                           ready=True,
                           best_model=metadata["best_model_name"],
                           all_results=all_res,
                           sorted_models=sorted_models,
                           best_metrics=metadata["best_model_metrics"])


@app.route("/about")
def about():
    """About page — methodology, limitations."""
    return render_template("about.html")


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    """Serve generated plot images from the plots/ directory."""
    plots_dir = BASE_DIR / "plots"
    return send_from_directory(str(plots_dir), filename)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
