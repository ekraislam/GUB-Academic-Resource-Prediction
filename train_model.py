"""
train_model.py
--------------
Full ML training pipeline for GUB Academic AI.

Steps:
  1. Load dataset
  2. Preprocessing (encode categoricals, scale numerics)
  3. Train 6 ML models
  4. Evaluate each model (accuracy, precision, recall, F1)
  5. Select best model by F1-score
  6. Generate visualisation plots
  7. Save best model + preprocessors via Joblib

NOTE: Dataset is SYNTHETIC / REPRESENTATIVE — not real GUB institutional data.
"""

import os
import sys
import io
import json
import warnings

# Force UTF-8 output so Unicode characters print correctly on all platforms
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Models ────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "data" / "university_academic_resource_dataset.csv"
MODELS_DIR  = BASE_DIR / "models"
PLOTS_DIR   = BASE_DIR / "plots"
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ─── 1. Load data ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  GUB Academic AI — Model Training Pipeline")
print("=" * 60)

if not DATA_PATH.exists():
    print("[WARN] Dataset not found. Running generate_dataset.py ...")
    import generate_dataset  # noqa: F401  (side-effect: creates the CSV)

df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Dataset loaded -> {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"     Target distribution:\n{df['target_resource_demand'].value_counts()}\n")

# ─── 2. Preprocessing ─────────────────────────────────────────────────────────
# Drop columns that are identifiers, not features
FEATURES = [
    "department", "course_type", "credit", "semester",
    "previous_enrollment", "previous_sections", "previous_pass_rate",
    "course_popularity", "lab_required", "lab_capacity", "room_capacity",
    "instructor_count", "class_duration", "previous_resource_usage",
    "registration_growth_rate"
]
TARGET = "target_resource_demand"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Categorical columns to label-encode
CAT_COLS = ["department", "course_type", "semester"]
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y_enc = target_encoder.fit_transform(y)          # LOW→0 MEDIUM→1 HIGH→2 (alphabetical)
label_names = list(target_encoder.classes_)      # ['HIGH', 'LOW', 'MEDIUM']

# ─── 3. Train / Test split ────────────────────────────────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Scale features: fit ONLY on training data to prevent data leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

print(f"Train size : {len(X_train):,}    Test size : {len(X_test):,}\n")

# ─── 4. Model definitions ─────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes":         GaussianNB(),
    "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
}

# ─── 5. Train & Evaluate ──────────────────────────────────────────────────────
results = {}
best_model_name = None
best_f1 = -1
best_model_obj = None
best_predictions = None

print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 65)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results[name] = {
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1_score":  round(f1, 4),
    }

    print(f"{name:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

    if f1 > best_f1:
        best_f1          = f1
        best_model_name  = name
        best_model_obj   = model
        best_predictions = y_pred

print("-" * 65)
print(f"\n[BEST] Best model : {best_model_name}  (F1 = {best_f1:.4f})\n")

# ─── 6. Visualisations ────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
ACCENT   = "#6C63FF"
PALETTE  = ["#6C63FF", "#43B89C", "#F7B731"]

# ── 6a. Demand distribution ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
counts = df[TARGET].value_counts()
bars = ax.bar(counts.index, counts.values, color=PALETTE, edgecolor="white", linewidth=1.2)
ax.set_title("Resource Demand Distribution", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Demand Level", fontsize=12)
ax.set_ylabel("Number of Courses", fontsize=12)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "demand_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6b. Enrollment distribution ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["previous_enrollment"], bins=30, color=ACCENT, edgecolor="white", linewidth=0.8)
ax.set_title("Previous Enrollment Distribution", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Previous Enrollment", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "enrollment_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6c. Demand by department ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
dept_demand = df.groupby(["department", TARGET]).size().unstack(fill_value=0)
dept_demand.plot(kind="bar", ax=ax, color=PALETTE, edgecolor="white", linewidth=0.8)
ax.set_title("Resource Demand by Department", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Department", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.legend(title="Demand", loc="upper right")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "demand_by_department.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6d. Course popularity distribution ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="course_popularity", palette="viridis", ax=ax)
ax.set_title("Course Popularity Distribution", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Course Popularity Score (1–10)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "course_popularity.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6e. Lab requirement distribution ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
lab_counts = df["lab_required"].value_counts()
ax.pie(lab_counts.values, labels=["No Lab", "Lab Required"][::-1],
       colors=["#43B89C", "#6C63FF"], autopct="%1.1f%%",
       startangle=140, wedgeprops=dict(edgecolor="white", linewidth=1.5))
ax.set_title("Lab Requirement Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "lab_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6f. Model accuracy comparison ────────────────────────────────────────────
model_names = list(results.keys())
accuracies  = [results[m]["accuracy"] for m in model_names]
f1_scores   = [results[m]["f1_score"] for m in model_names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
bars = axes[0].barh(model_names, accuracies, color=ACCENT, edgecolor="white")
axes[0].set_xlim(0, 1.05)
axes[0].set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Accuracy")
for bar, val in zip(bars, accuracies):
    axes[0].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=9)

# F1-score
bars2 = axes[1].barh(model_names, f1_scores, color="#F7B731", edgecolor="white")
axes[1].set_xlim(0, 1.05)
axes[1].set_title("Model F1-Score Comparison", fontsize=13, fontweight="bold")
axes[1].set_xlabel("F1-Score")
for bar, val in zip(bars2, f1_scores):
    axes[1].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=9)

plt.suptitle("ML Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6g. Confusion matrix for best model ──────────────────────────────────────
cm = confusion_matrix(y_test, best_predictions)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            linewidths=0.5, linecolor="white", ax=ax)
ax.set_title(f"Confusion Matrix — {best_model_name}", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

print("[OK] Visualisations saved ->", PLOTS_DIR)

# ─── 7. Save artefacts ────────────────────────────────────────────────────────
joblib.dump(best_model_obj,  MODELS_DIR / "best_model.pkl")
joblib.dump(scaler,          MODELS_DIR / "scaler.pkl")
joblib.dump(encoders,        MODELS_DIR / "encoders.pkl")
joblib.dump(target_encoder,  MODELS_DIR / "target_encoder.pkl")

# Save metadata (model name, metrics, feature list) as JSON for the Flask app
metadata = {
    "best_model_name": best_model_name,
    "best_model_metrics": results[best_model_name],
    "all_model_results": results,
    "features": FEATURES,
    "label_names": label_names,
    "dataset_shape": list(df.shape),
    "demand_counts": df[TARGET].value_counts().to_dict(),
}
with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("[OK] Model artefacts saved ->", MODELS_DIR)
print("\n" + "=" * 60)
print("  Training complete!")
print(f"  Best Model : {best_model_name}")
print(f"  Accuracy   : {results[best_model_name]['accuracy']:.4f}")
print(f"  Precision  : {results[best_model_name]['precision']:.4f}")
print(f"  Recall     : {results[best_model_name]['recall']:.4f}")
print(f"  F1-Score   : {results[best_model_name]['f1_score']:.4f}")
print("=" * 60)
