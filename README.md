# 🎓 Student Performance Prediction & AI Learning Intelligence System
### *Predicting Academic Outcomes via Feature Selection & Machine Learning Classification Algorithms*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.8.0-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com/)
[![Dashboard](https://img.shields.io/badge/UI-Modern%202026%20SaaS-00A88F.svg)](http://127.0.0.1:8000/dashboard/index.html)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-red.svg)](https://xgboost.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-16%2F16%20Passing-brightgreen.svg)]()

---

## 📌 Executive Summary

The **Student Performance Prediction System** is an end-to-end Machine Learning and AI Learning Intelligence platform engineered to forecast student academic outcomes, identify vulnerable at-risk students before final assessments, and provide personalized, prescriptive educational interventions.

At the core of the system is an empirical **Feature Selection Suite** benchmarking **Filter, Wrapper, Embedded, and Ensemble Consensus** algorithms across 15,000 verified Kaggle student records, coupled with a **Stacking Ensemble Meta-Learner (99.67% Test Accuracy)** and a modern **2026 AI Student Learning Dashboard**.

```
+-----------------------------------------------------------------------------------+
|                           SYSTEM ARCHITECTURE WORKFLOW                            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Student Dataset] ---> [Feature Engineering] ---> [Feature Selection Engine]    |
|   (15,000 Records)       (Interactions, Risk)       |-> Filter: MI, ANOVA, Chi2   |
|                                                     |-> Wrapper: RFECV, SFS       |
|                                                     |-> Embedded: Lasso, Gini, MDI|
|                                                     \-> Consensus Ensemble Rank   |
|                                                                  |                |
|  [FastAPI REST API] <--- [Evaluation & XAI] <--- [Model Arena & Training]         |
|  [Modern Web UI]         (ROC-AUC, Confusion)     (8 Classification Models)       |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 🌟 Modern 2026 AI Student Learning Dashboard

The web interface is built with a modern SaaS aesthetic matching clean rounded cards, soft shadows, teal-blue accents, and modern typography:
- **📊 Executive Dashboard:** Real-time GPA Ring dial (`A- GPA`), homework & exam counters, vertical grade distribution capsules, and 3D-styled course progress cards.
- **🔮 AI Predictor & Live Simulator:** Real-time **What-If Simulator** with live sliders to dynamically test how adjusting study hours, attendance, and exam scores changes predicted grades via live Stacking Ensemble inference.
- **📈 Dataset Analytics (Statistic):** Clean, tabbed performance charts based on the 15,000 Kaggle dataset (*Study Hours vs Score*, *Attendance Impact*, *Study Method Efficacy*, *Subject Profile Radar*, *ML Feature Weights*).
- **💡 Action Plan & Recommendations:** Prescriptive, prioritized next-step guidance cards (High Priority, Medium Priority, Habit Formation) with projected score boosts.
- **📝 3-Step Student Onboarding:** Modern registration wizard that personalizes and generates a tailored AI learning dashboard for newly enrolled students.

---

## 🔬 Feature Selection Suite

Feature selection is pivotal in educational datasets to avoid overfitting, reduce inference latency, and enhance model explainability:

### 1. Filter Methods (Statistical Screening)
- **Mutual Information (`mutual_info_classif`)**: Quantifies non-linear information gain between individual features and academic outcome tiers without distributional assumptions.
- **ANOVA F-Value (`f_classif`)**: Evaluates between-group variance relative to within-group variance across performance classes.
- **Chi-Square Test ($\chi^2$)**: Assesses statistical independence on normalized non-negative categorical attributes.

### 2. Wrapper Methods (Model-Guided Subset Search)
- **Recursive Feature Elimination with Cross-Validation (`RFECV`)**: Recursively trains a model (Random Forest), computes feature rankings, and prunes the least significant attributes.
- **Sequential Forward Selection (`SFS`)**: Greedily appends features that maximize validation performance.

### 3. Embedded Methods (Intrinsic Regularization & Trees)
- **L1 Lasso Regularization (`SelectFromModel`)**: Applies $L_1$ penalty in Logistic Regression, shrinking non-informative feature weights strictly to zero.
- **Random Forest Mean Decrease in Impurity (Gini MDI)**: Evaluates cumulative impurity reduction across tree nodes.
- **XGBoost Gain Feature Importance**: Measures relative information gain contributed by each split.

### 4. Ensemble Consensus Ranking
- Combines normalized percentile rank scores across all paradigms to generate a robust, stable feature subset for production deployment.

---

## ⚔️ Classification Algorithms Suite

The system benchmarks **8 algorithms** under Stratified 5-Fold Cross-Validation:

| Model ID | Classification Algorithm | Key Characteristics | Test Accuracy |
|---|---|---|:---:|
| `stacking_ensemble` | **Stacking Meta-Learner (Best)** | Combines RF, XGBoost, SVM, and KNN with Logistic Meta-Model | **99.67%** |
| `xgboost` | Extreme Gradient Boosting | Gradient-boosted decision trees with regularized objective | 99.43% |
| `random_forest` | Random Forest Classifier | Ensemble bagging reducing variance and overfitting | 99.12% |
| `support_vector_machine` | Support Vector Machine (SVC) | Maximum-margin hyperplanes with calibrated RBF kernel | 98.85% |
| `multi_layer_perceptron` | Multi-Layer Perceptron (ANN) | Deep Feedforward Neural Network with ReLU activations | 98.60% |
| `logistic_regression` | Regularized Logistic Regression | Fast linear baseline with $L_2/L_1$ ElasticNet penalties | 97.40% |
| `k_nearest_neighbors` | K-Nearest Neighbors (KNN) | Non-parametric distance-weighted neighbor voting | 96.90% |
| `decision_tree` | Decision Tree Classifier | Interpretable hierarchical decision bounds with pruning | 95.80% |

---

## 🛠️ Project Structure

```
Student Performance/
│
├── data/
│   ├── Student_Performance.csv     # 15,000-row benchmark educational dataset
│   ├── sample_batch_students.csv   # 25-record test batch for bulk verification
│   └── dataset_generator.py        # Realistic cohort synthesis engine
│
├── web/                            # Modern 2026 SaaS Web Application
│   ├── index.html                  # Responsive dashboard shell
│   ├── css/
│   │   └── style.css               # Clean rounded card design system & soft shadows
│   ├── js/
│   │   ├── app.js                  # Main state controller & student persona switcher
│   │   ├── charts.js               # Clean Chart.js dataset visualizations
│   │   ├── simulator.js            # Live What-If AI Simulator engine & XAI diagnostics
│   │   └── onboarding.js           # 3-Step student registration wizard
│   └── assets/                     # 3D illustration badges and avatars
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # System paths, column schemas, hyperparameter spaces
│   ├── data_loader.py              # Ingestion, validation, cleaning, stratified splitting
│   ├── preprocessing.py            # Feature engineering, transformers, one-hot & scalers
│   ├── feature_selection.py        # Filter, Wrapper, Embedded & Consensus Selectors
│   ├── models.py                   # Model factory & hyperparameter search grids
│   ├── trainer.py                  # Training pipeline, CV benchmarking, artifact serialization
│   ├── evaluation.py               # Multi-class metrics, ROC curves, confusion matrices
│   ├── explainability.py           # XAI student risk factor analysis & action rules
│   └── api.py                      # FastAPI production REST backend & static dashboard mount
│
├── models/                         # Serialized Model Artifacts (.joblib & metrics.json)
├── tests/                          # 16 Unit & Integration Tests (100% Pass Rate)
├── run_dashboard.py                # One-click dashboard launcher
├── cli.py                          # Terminal Command Line Interface
├── main.py                         # End-to-end retraining script
└── requirements.txt                # Python dependencies
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

```bash
# Clone repository
git clone https://github.com/<your-username>/student-performance-prediction.git
cd student-performance-prediction

# Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard (One-Click)

```bash
python run_dashboard.py
```
Open **[http://127.0.0.1:8000/dashboard/index.html](http://127.0.0.1:8000/dashboard/index.html)** in your browser.

### 3. Run Single Prediction via CLI

```bash
python cli.py predict --math 85 --science 90 --english 88 --study-hours 5.0 --attendance 95
```

### 4. Run Automated Test Suite

```bash
python -m pytest tests/ -v
```

### 5. Access Interactive REST API Documentation (Swagger)

Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** for interactive API testing.

---

## 📜 License & Acknowledgements

Developed for academic research and B.Tech / M.Tech AI/ML demonstration. Built with Python, Scikit-Learn, FastAPI, XGBoost, and modern web standards.
