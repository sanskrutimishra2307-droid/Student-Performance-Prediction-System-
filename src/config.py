"""
Central Configuration & Schema Definitions for Student Performance Prediction System.
"""

from pathlib import Path
from typing import List, Dict, Any

# Root Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

DEFAULT_DATASET_PATH = DATA_DIR / "Student_Performance.csv"
SAMPLE_BATCH_PATH = DATA_DIR / "sample_batch_students.csv"

# Model Artifact Paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
FEATURE_SELECTOR_PATH = MODELS_DIR / "feature_selector.joblib"
METRICS_REPORT_PATH = MODELS_DIR / "metrics_summary.json"

# Column Definitions
ID_COLUMN = "student_id"
RAW_TARGET_COLUMN = "final_grade"
RAW_SCORE_COLUMN = "overall_score"

NUMERICAL_FEATURES = [
    "age",
    "study_hours",
    "attendance_percentage",
    "math_score",
    "science_score",
    "english_score"
]

# Features that can be used when predicting BEFORE exams (No test scores yet)
PRE_EXAM_NUMERICAL_FEATURES = [
    "age",
    "study_hours",
    "attendance_percentage"
]

CATEGORICAL_FEATURES = [
    "gender",
    "school_type",
    "parent_education",
    "internet_access",
    "travel_time",
    "extra_activities",
    "study_method"
]

ORDINAL_MAPPINGS = {
    "parent_education": {
        "no formal": 0,
        "high school": 1,
        "diploma": 2,
        "graduate": 3,
        "post graduate": 4,
        "phd": 5
    },
    "travel_time": {
        "<15 min": 0,
        "15-30 min": 1,
        "30-60 min": 2,
        ">60 min": 3
    },
    "internet_access": {
        "no": 0,
        "yes": 1
    },
    "extra_activities": {
        "no": 0,
        "yes": 1
    }
}

# Target Classification Schemes
TARGET_SCHEMES = {
    "3-tier": {
        "name": "3-Tier Performance (Distinction / Pass / At-Risk)",
        "mapping": {
            "a": "Distinction",
            "b": "Distinction",
            "c": "Pass",
            "d": "Pass",
            "e": "At-Risk",
            "f": "At-Risk"
        },
        "classes": ["At-Risk", "Pass", "Distinction"]
    },
    "binary": {
        "name": "Binary Classification (Pass / Fail)",
        "mapping": {
            "a": "Pass",
            "b": "Pass",
            "c": "Pass",
            "d": "Pass",
            "e": "Fail",
            "f": "Fail"
        },
        "classes": ["Fail", "Pass"]
    },
    "multi-grade": {
        "name": "6-Tier Letter Grades (A / B / C / D / E / F)",
        "mapping": {
            "a": "A",
            "b": "B",
            "c": "C",
            "d": "D",
            "e": "E",
            "f": "F"
        },
        "classes": ["F", "E", "D", "C", "B", "A"]
    }
}

DEFAULT_TARGET_SCHEME = "3-tier"

# Feature Selection Algorithms Supported
FEATURE_SELECTION_METHODS = [
    "mutual_info",
    "anova_f",
    "chi2",
    "rfecv",
    "sfs",
    "lasso_l1",
    "random_forest_mdi",
    "xgboost_mdi",
    "consensus"
]

DEFAULT_NUM_FEATURES = 8

# Random Seed for Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
