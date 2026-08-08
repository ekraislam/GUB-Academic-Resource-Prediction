# GUB Academic AI — Academic Resource Demand Prediction & Planning System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Status-Prototype-yellow" alt="Status">
  <img src="https://img.shields.io/badge/Dataset-Synthetic-red" alt="Dataset">
</p>

---

## 📌 Project Overview

**GUB Academic AI** is a complete machine learning web application designed as a prototype for **Green University of Bangladesh (GUB)**. It predicts the level of academic resource demand — **LOW**, **MEDIUM**, or **HIGH** — for university courses, based on enrollment trends, course characteristics, and historical resource usage.

The system helps academic planners estimate:
- Expected student demand
- Recommended number of sections
- Classroom capacity requirements
- Lab requirements
- Overall academic resource planning priority

---

## ⚠️ Important Dataset Disclaimer

> **This application uses a completely SYNTHETIC / REPRESENTATIVE dataset.**
>
> The data was programmatically generated to simulate realistic academic patterns and does **NOT** contain any real, confidential, or official Green University of Bangladesh institutional data. No student names, IDs, phone numbers, or personally identifiable information exist anywhere in this system.
>
> The dataset is intended solely as a demonstration of the machine learning methodology. Results should not be interpreted as actual GUB institutional analysis.

---

## 🎯 Problem Statement

University academic departments frequently struggle to anticipate resource demands for upcoming semesters. Misallocations lead to:
- Overcrowded classrooms or underutilized resources
- Insufficient lab availability
- Instructor shortages
- Poor student experience

This system provides a data-driven approach to proactively predict and plan resource allocation.

---

## 🚀 Features

- ✅ Multi-class ML classification (LOW / MEDIUM / HIGH demand)
- ✅ Six ML algorithms trained and compared
- ✅ Automatic best model selection by F1-score
- ✅ Interactive prediction form with 16 input features
- ✅ Rule-based planning recommendations
- ✅ Model performance comparison with charts
- ✅ Generated data visualizations (matplotlib/seaborn)
- ✅ Responsive dark academic UI
- ✅ 5-page Flask web application
- ✅ Vercel deployment ready

---

## 🧠 ML Methodology

### Pipeline

```
Synthetic Dataset (5,000 records)
    ↓
Data Preprocessing
  • LabelEncoder for categoricals (department, course_type, semester)
  • StandardScaler for all numeric features
  • 80/20 stratified train/test split
    ↓
Feature Engineering (16 features)
    ↓
Model Training (6 algorithms)
    ↓
Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
    ↓
Best Model Selection (highest weighted F1-score)
    ↓
Artifact Saving (Joblib: model + scaler + encoders)
    ↓
Flask Web Application → Prediction → Recommendation
```

---

## 📊 Dataset Description

| Field | Type | Description |
|---|---|---|
| `department` | Category | Academic department (CSE, EEE, BBA, …) |
| `course_type` | Category | Theory / Lab / Theory+Lab / Seminar |
| `credit` | Integer | Credit hours (1–4) |
| `semester` | Category | Spring / Summer / Fall |
| `previous_enrollment` | Integer | Students in last offering |
| `previous_sections` | Integer | Sections last semester |
| `previous_pass_rate` | Float | % who passed |
| `course_popularity` | Integer | Popularity score (1–10) |
| `lab_required` | Binary | 1 = lab needed, 0 = no lab |
| `lab_capacity` | Integer | Max lab seats |
| `room_capacity` | Integer | Classroom capacity |
| `instructor_count` | Integer | Number of instructors |
| `class_duration` | Integer | Duration in minutes |
| `previous_resource_usage` | Float | Historical resource utilisation (0–100) |
| `registration_growth_rate` | Float | % growth in registrations |
| `target_resource_demand` | **Target** | **LOW / MEDIUM / HIGH** |

**Dataset size:** ~5,000 records, 17 columns (16 features + 1 target)

---

## 🤖 Algorithms Used

| # | Algorithm | Notes |
|---|---|---|
| 1 | Logistic Regression | Linear baseline classifier |
| 2 | K-Nearest Neighbors | Instance-based, k=7 |
| 3 | Naive Bayes | Probabilistic (Gaussian) |
| 4 | Decision Tree | Non-linear, max_depth=10 |
| 5 | Random Forest | Ensemble, 100 estimators |
| 6 | SVM | RBF kernel, probability=True |

---

## 📈 Evaluation Metrics

- **Accuracy** — Overall correct prediction rate
- **Precision** — Correct positive rate (weighted)
- **Recall** — True positive capture rate (weighted)
- **F1-Score** — Harmonic mean of precision & recall (primary selection metric)
- **Confusion Matrix** — Per-class breakdown

---

## 🖥️ Pages

| Route | Page | Description |
|---|---|---|
| `/` | Dashboard | KPI cards, model summary, visualizations |
| `/predict` | Prediction Form | Enter 16 features, submit for prediction |
| `/result` | Prediction Result | Demand level, planning estimates, recommendations |
| `/model-performance` | Model Performance | Comparison table, charts, methodology |
| `/about` | About | Project info, limitations, tech stack |

---

## 🖼️ Screenshots

> *(Add screenshots here after running the application)*

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or later
- pip

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/gub-academic-ai.git
cd gub-academic-ai
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ How to Train the Model

### Step 1: Generate the synthetic dataset
```bash
python generate_dataset.py
```
This creates `data/university_academic_resource_dataset.csv` (~5,000 records).

### Step 2: Train all models and generate plots
```bash
python train_model.py
```
This will:
- Train 6 ML algorithms
- Evaluate each model on a 20% test set
- Save the best model to `models/`
- Generate plots in `plots/`
- Print a performance summary

---

## 🌐 How to Run Flask Locally

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## ☁️ Deployment (Vercel)

1. Push this project to a GitHub repository.
2. Go to [vercel.com](https://vercel.com) → **Import Project**.
3. Select your GitHub repo.
4. Vercel auto-detects Python/Flask via `vercel.json`.
5. Deploy.

> **Note:** Vercel's serverless environment may require pre-committing the trained model files (`models/`) since training cannot run at build time.

---

## ⚠️ Limitations

1. Dataset is **synthetic** — results may differ significantly with real GUB data.
2. No temporal/time-series modelling (semester sequence is not exploited).
3. Hyperparameters use sensible defaults but are not exhaustively tuned.
4. Single train/test split (no cross-validation).
5. The recommendation engine is **rule-based**, not a learned model.
6. No authentication or admin panel for data management.

---

## 🔮 Future Improvements

- Integrate actual GUB institutional data (with proper consent & anonymization).
- Add cross-validation and hyperparameter tuning (GridSearchCV / Optuna).
- Deploy SHAP/LIME explainability for per-prediction feature importance.
- Implement time-series forecasting for multi-semester planning horizons.
- Build an admin panel for uploading new data and re-training.
- Add export-to-PDF/CSV for planning reports.
- Deploy a REST API endpoint for integration with GUB ERP systems.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn, pandas, numpy, joblib |
| Visualization | matplotlib, seaborn, Chart.js |
| Frontend | HTML5, CSS3, JavaScript, Bootstrap Icons |
| Deployment | Vercel, gunicorn |

---

## 📄 License

This project is intended for educational and academic demonstration purposes.

---

## 👤 Author

Developed as a university prototype for Green University of Bangladesh (GUB).

---

> **Reminder:** This system uses a synthetic dataset. It is a prototype and does not represent official GUB institutional data or policy.
