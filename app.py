"""
app.py
------
Flask web application for GUB Academic AI.
Serves five pages: Dashboard, Predict, Result, Model Performance, About.

Run:  python app.py
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory

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
    """Dashboard — summary statistics."""
    if metadata is None:
        return render_template("index.html", ready=False)

    dc = metadata.get("demand_counts", {})
    total = sum(dc.values())
    all_res = metadata.get("all_model_results", {})
    sorted_models = sorted(all_res.items(), key=lambda x: x[1]["f1_score"], reverse=True)

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
            "registration_growth_rate": form.get("registration_growth_rate"),
        },
    }
    return redirect(url_for("result"))


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
