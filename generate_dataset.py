"""
generate_dataset.py
-------------------
Generates a realistic SYNTHETIC academic resource dataset for GUB Academic AI.
NOTE: This dataset is entirely synthetic / representative and does NOT contain
any real or confidential Green University of Bangladesh institutional data.
"""

import sys
import io

# Force UTF-8 output so Unicode characters print correctly on all platforms
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import os
from pathlib import Path

# ─── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)
N = 5000  # number of records

# ─── Domain lists ────────────────────────────────────────────────────────────
departments = [
    "CSE", "EEE", "BBA", "LAW", "English",
    "Economics", "Architecture", "Pharmacy", "Civil", "Textile"
]

course_types = ["Theory", "Lab", "Theory+Lab", "Seminar"]
semesters    = ["Spring", "Summer", "Fall"]

# ─── Helper: weighted random choice ──────────────────────────────────────────
def wchoice(options, weights, size):
    return np.random.choice(options, size=size, p=np.array(weights) / sum(weights))

# ─── Base random features ─────────────────────────────────────────────────────
department        = np.random.choice(departments, N)
course_type       = wchoice(course_types, [50, 20, 25, 5], N)
semester          = np.random.choice(semesters, N)
credit            = np.random.choice([1, 2, 3, 4], N, p=[0.05, 0.15, 0.55, 0.25])

# Lab-required logic: Lab / Theory+Lab courses and some CSE/EEE almost always need lab
lab_required_prob = np.where(
    np.isin(course_type, ["Lab", "Theory+Lab"]), 0.95,
    np.where(np.isin(department, ["CSE", "EEE"]), 0.45, 0.15)
)
lab_required = (np.random.rand(N) < lab_required_prob).astype(int)

# Enrollment: correlated with department popularity
dept_enrollment_mean = {
    "CSE": 90, "EEE": 75, "BBA": 85, "LAW": 55, "English": 50,
    "Economics": 60, "Architecture": 40, "Pharmacy": 65, "Civil": 70, "Textile": 55
}
dept_std = {d: m * 0.25 for d, m in dept_enrollment_mean.items()}

prev_enrollment = np.array([
    max(10, int(np.random.normal(dept_enrollment_mean[d], dept_std[d])))
    for d in department
])

# Sections: roughly enrollment / 40, with noise
prev_sections = np.clip(np.round(prev_enrollment / 40 + np.random.normal(0, 0.4, N)).astype(int), 1, 8)

# Pass rate: 55–98%
prev_pass_rate = np.clip(np.random.normal(78, 12, N), 45, 98).round(1)

# Course popularity: 1–10 score, correlated slightly with enrollment
course_popularity = np.clip(
    np.round(prev_enrollment / 15 + np.random.normal(0, 1.2, N)), 1, 10
).astype(int)

# Lab capacity: only meaningful when lab_required; else 0
lab_capacity = np.where(lab_required == 1,
                        np.random.choice([20, 25, 30, 40, 50], N),
                        0)

# Room capacity: should be ≥ enrollment / sections
room_capacity_base = (prev_enrollment / np.maximum(prev_sections, 1)).astype(int)
room_capacity = np.clip(
    room_capacity_base + np.random.randint(-5, 20, N),
    20, 200
)

# Instructor count: 1–5
instructor_count = np.clip(
    np.random.choice([1, 2, 3, 4, 5], N, p=[0.25, 0.35, 0.25, 0.10, 0.05]),
    1, 5
)

# Class duration in minutes
class_duration = np.random.choice([50, 75, 90, 120, 150, 180], N,
                                   p=[0.20, 0.25, 0.20, 0.15, 0.10, 0.10])

# Previous resource usage (0–100 scale)
prev_resource_usage = np.clip(
    np.random.normal(60, 20, N), 10, 100
).round(1)

# Registration growth rate (-20% to +40%)
reg_growth_rate = np.clip(
    np.random.normal(8, 12, N), -20, 45
).round(2)

# Course code (synthetic)
dept_code_map = {
    "CSE": "CSE", "EEE": "EEE", "BBA": "BBA", "LAW": "LAW", "English": "ENG",
    "Economics": "ECO", "Architecture": "ARC", "Pharmacy": "PHR", "Civil": "CVE", "Textile": "TXE"
}
course_number = np.random.choice(range(100, 500), N)
course_code = np.array([f"{dept_code_map[d]}{n}" for d, n in zip(department, course_number)])

# ─── Target: Resource Demand ──────────────────────────────────────────────────
# Compute a demand score from features (logically consistent)
demand_score = (
    (prev_enrollment / 100) * 35          # enrollment is strongest signal
    + (reg_growth_rate / 45) * 20         # growth matters
    + (course_popularity / 10) * 15       # popularity
    + lab_required * 10                   # labs need more resources
    + ((100 - prev_pass_rate) / 100) * 8  # low pass rate → more support needed
    + (prev_resource_usage / 100) * 7     # prior usage
    + (credit / 4) * 5                    # higher credit = more resources
    + np.random.normal(0, 5, N)           # noise
)

# Normalise to 0–100
score_min, score_max = demand_score.min(), demand_score.max()
demand_score_norm = (demand_score - score_min) / (score_max - score_min) * 100

# Bin into LOW / MEDIUM / HIGH
target_resource_demand = np.where(
    demand_score_norm < 33, "LOW",
    np.where(demand_score_norm < 67, "MEDIUM", "HIGH")
)

# ─── Assemble DataFrame ───────────────────────────────────────────────────────
df = pd.DataFrame({
    "department":              department,
    "course_code":             course_code,
    "course_type":             course_type,
    "credit":                  credit,
    "semester":                semester,
    "previous_enrollment":     prev_enrollment,
    "previous_sections":       prev_sections,
    "previous_pass_rate":      prev_pass_rate,
    "course_popularity":       course_popularity,
    "lab_required":            lab_required,
    "lab_capacity":            lab_capacity,
    "room_capacity":           room_capacity,
    "instructor_count":        instructor_count,
    "class_duration":          class_duration,
    "previous_resource_usage": prev_resource_usage,
    "registration_growth_rate":reg_growth_rate,
    "target_resource_demand":  target_resource_demand,
})

# ─── Save ─────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent / "data"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "university_academic_resource_dataset.csv"
df.to_csv(out_path, index=False)

print(f"[OK] Dataset saved -> {out_path}")
print(f"     Shape       : {df.shape}")
print(f"     Target dist :\n{df['target_resource_demand'].value_counts()}")
