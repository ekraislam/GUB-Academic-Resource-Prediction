"""
audit_predictions.py
--------------------
Script to execute and document 5 distinct realistic test predictions for audit.
"""

import sys
import io
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import app as flask_app

# Force UTF-8 output so Unicode characters print correctly on all platforms
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

model = joblib.load(MODELS_DIR / "best_model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
encoders = joblib.load(MODELS_DIR / "encoders.pkl")
target_encoder = joblib.load(MODELS_DIR / "target_encoder.pkl")
with open(MODELS_DIR / "metadata.json") as f:
    metadata = json.load(f)

test_cases = [
    {
        "name": "Test 1: High enrollment + high growth + high popularity",
        "form": {
            "department": "CSE", "course_code": "CSE401", "course_type": "Theory",
            "credit": "3", "semester": "Fall", "previous_enrollment": "180",
            "previous_sections": "3", "previous_pass_rate": "60", "course_popularity": "9",
            "lab_required": "0", "lab_capacity": "0", "room_capacity": "50",
            "instructor_count": "2", "class_duration": "90", "previous_resource_usage": "90",
            "registration_growth_rate": "35"
        }
    },
    {
        "name": "Test 2: Low enrollment + low growth + low popularity",
        "form": {
            "department": "English", "course_code": "ENG101", "course_type": "Theory",
            "credit": "3", "semester": "Spring", "previous_enrollment": "20",
            "previous_sections": "1", "previous_pass_rate": "90", "course_popularity": "2",
            "lab_required": "0", "lab_capacity": "0", "room_capacity": "60",
            "instructor_count": "1", "class_duration": "90", "previous_resource_usage": "20",
            "registration_growth_rate": "-15"
        }
    },
    {
        "name": "Test 3: Medium enrollment + medium growth",
        "form": {
            "department": "BBA", "course_code": "MGT201", "course_type": "Theory",
            "credit": "3", "semester": "Summer", "previous_enrollment": "70",
            "previous_sections": "2", "previous_pass_rate": "78", "course_popularity": "5",
            "lab_required": "0", "lab_capacity": "0", "room_capacity": "60",
            "instructor_count": "2", "class_duration": "90", "previous_resource_usage": "55",
            "registration_growth_rate": "8"
        }
    },
    {
        "name": "Test 4: High enrollment + lab required",
        "form": {
            "department": "EEE", "course_code": "EEE305", "course_type": "Theory+Lab",
            "credit": "4", "semester": "Fall", "previous_enrollment": "160",
            "previous_sections": "4", "previous_pass_rate": "65", "course_popularity": "8",
            "lab_required": "1", "lab_capacity": "30", "room_capacity": "45",
            "instructor_count": "3", "class_duration": "180", "previous_resource_usage": "85",
            "registration_growth_rate": "25"
        }
    },
    {
        "name": "Test 5: Low enrollment + lab required",
        "form": {
            "department": "Textile", "course_code": "TXE102", "course_type": "Lab",
            "credit": "2", "semester": "Spring", "previous_enrollment": "25",
            "previous_sections": "1", "previous_pass_rate": "85", "course_popularity": "3",
            "lab_required": "1", "lab_capacity": "25", "room_capacity": "40",
            "instructor_count": "1", "class_duration": "120", "previous_resource_usage": "40",
            "registration_growth_rate": "2"
        }
    }
]

print("=" * 80)
print("AUDIT PREDICTION RUNNER")
print("=" * 80)

for tc in test_cases:
    form = tc["form"]
    dept_enc = encoders["department"].transform([form["department"]])[0]
    ctype_enc = encoders["course_type"].transform([form["course_type"]])[0]
    sem_enc = encoders["semester"].transform([form["semester"]])[0]

    feature_dict = {
        "department":              dept_enc,
        "course_type":             ctype_enc,
        "credit":                  int(form["credit"]),
        "semester":                sem_enc,
        "previous_enrollment":     int(form["previous_enrollment"]),
        "previous_sections":       int(form["previous_sections"]),
        "previous_pass_rate":      float(form["previous_pass_rate"]),
        "course_popularity":       int(form["course_popularity"]),
        "lab_required":            int(form["lab_required"]),
        "lab_capacity":            int(form["lab_capacity"]),
        "room_capacity":           int(form["room_capacity"]),
        "instructor_count":        int(form["instructor_count"]),
        "class_duration":          int(form["class_duration"]),
        "previous_resource_usage": float(form["previous_resource_usage"]),
        "registration_growth_rate":float(form["registration_growth_rate"]),
    }
    df_feat = pd.DataFrame([feature_dict])[metadata["features"]]
    df_scaled = scaler.transform(df_feat)
    
    pred_enc = model.predict(df_scaled)[0]
    demand = target_encoder.inverse_transform([pred_enc])[0]
    
    try:
        proba = model.predict_proba(df_scaled)[0]
        prob_dict = {lbl: f"{p*100:.1f}%" for lbl, p in zip(target_encoder.classes_, proba)}
    except Exception:
        prob_dict = {demand: "100%"}

    rec = flask_app.build_recommendation(demand, form)
    est_enr = flask_app.estimate_enrollment(form, demand)
    rec_sec = flask_app.recommend_sections(est_enr)

    print(f"\n--- {tc['name']} ---")
    print("Department:", form["department"], "| Course Code:", form["course_code"])
    print("Preprocessed feature array:\n", df_feat.values[0])
    print("Model Prediction:", demand)
    print("Model Probabilities:", prob_dict)
    print(f"Estimates: Expected Enrollment={est_enr}, Rec. Sections={rec_sec}")
    print("Recommendation:", rec)

print("=" * 80)
