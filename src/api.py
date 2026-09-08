"""
FastAPI Production REST API Backend for Student Performance Prediction & Learning Intelligence.
Provides standardized RESTful endpoints for real-time inference, batch scoring, model benchmarking, explainability, and static dashboard serving.
"""

import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from src.trainer import ModelTrainer
from src.explainability import ExplainabilityEngine
from src.config import TARGET_SCHEMES, DEFAULT_TARGET_SCHEME, DEFAULT_DATASET_PATH

app = FastAPI(
    title="Student Performance Prediction & Learning Intelligence API",
    description="Production-grade Machine Learning API for student outcome classification and feature selection analysis.",
    version="2.0.0"
)

# Enable CORS for web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Dashboard Mount ---
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
if os.path.exists(WEB_DIR):
    app.mount("/dashboard", StaticFiles(directory=WEB_DIR, html=True), name="dashboard")

# --- Pydantic Data Contracts ---

class StudentInput(BaseModel):
    age: int = Field(16, ge=10, le=30, description="Student age in years")
    gender: str = Field("female", description="Gender (male, female, other)")
    school_type: str = Field("public", description="School type (public, private)")
    parent_education: str = Field("graduate", description="Parent education (no formal, high school, diploma, graduate, post graduate, phd)")
    study_hours: float = Field(4.5, ge=0.0, le=24.0, description="Daily self-study hours")
    attendance_percentage: float = Field(80.0, ge=0.0, le=100.0, description="Attendance percentage (0-100)")
    internet_access: str = Field("yes", description="Home internet access (yes, no)")
    travel_time: str = Field("<15 min", description="Daily travel time (<15 min, 15-30 min, 30-60 min, >60 min)")
    extra_activities: str = Field("yes", description="Participation in extracurriculars (yes, no)")
    study_method: str = Field("notes", description="Primary study method (notes, textbook, group study, online videos, coaching, mixed)")
    math_score: Optional[float] = Field(65.0, ge=0.0, le=100.0, description="Math score (optional if pre-exam)")
    science_score: Optional[float] = Field(68.0, ge=0.0, le=100.0, description="Science score (optional if pre-exam)")
    english_score: Optional[float] = Field(70.0, ge=0.0, le=100.0, description="English score (optional if pre-exam)")

class SinglePredictionResponse(BaseModel):
    status: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_name: str
    used_features: List[str]
    explanation: Dict[str, Any]

class BatchStudentInput(BaseModel):
    students: List[StudentInput]

class TrainRequest(BaseModel):
    feature_selection_method: str = Field("consensus", description="Feature selection method (consensus, mutual_info, anova_f, chi2, rfecv, lasso_l1, random_forest_mdi, xgboost_mdi, all_features)")
    n_features: int = Field(8, ge=2, le=25, description="Number of top features to select")
    target_scheme: str = Field("3-tier", description="Target classification scheme (3-tier, binary, multi-grade)")
    tune_hyperparameters: bool = Field(False, description="Whether to run grid search hyperparameter tuning")

# --- Endpoints ---

@app.get("/", tags=["General"])
def root():
    return {
        "service": "Student Performance Prediction API",
        "version": "2.0.0",
        "documentation": "/docs",
        "dashboard": "/dashboard/index.html",
        "status": "online"
    }

@app.get("/health", tags=["General"])
def health_check():
    try:
        model, preprocessor, _, metadata = ModelTrainer.load_trained_system()
        return {
            "status": "healthy",
            "model_loaded": True,
            "active_model": metadata.get("best_model_name", "Unknown"),
            "best_f1": metadata.get("best_f1_score", None)
        }
    except Exception as e:
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": str(e)
        }

@app.get("/api/v1/features", tags=["Metadata"])
def get_features_metadata():
    try:
        _, _, selector, metadata = ModelTrainer.load_trained_system()
        return {
            "selected_features": metadata.get("selected_features", []),
            "all_features": metadata.get("all_features", []),
            "classes": metadata.get("class_names", []),
            "target_scheme": metadata.get("target_scheme", DEFAULT_TARGET_SCHEME)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/personas", tags=["Metadata"])
def get_student_personas():
    """Returns curated realistic student personas based on the Kaggle dataset."""
    return {
        "personas": [
            {
                "id": "adam",
                "name": "Adam Chen",
                "role": "High Achiever",
                "avatar": "assets/avatar_adam.png",
                "grade": "A",
                "gpa": "A (91%)",
                "overall_score": 91.2,
                "data": {
                    "age": 17,
                    "gender": "male",
                    "school_type": "public",
                    "parent_education": "post graduate",
                    "study_hours": 5.5,
                    "attendance_percentage": 94.0,
                    "internet_access": "yes",
                    "travel_time": "<15 min",
                    "extra_activities": "yes",
                    "study_method": "coaching",
                    "math_score": 88.0,
                    "science_score": 93.0,
                    "english_score": 91.0
                },
                "homework_completed": 112,
                "homework_total": 120,
                "exams_completed": 11,
                "exams_total": 12
            },
            {
                "id": "michael",
                "name": "Michael Brown",
                "role": "Steady Performer",
                "avatar": "assets/avatar_michael.png",
                "grade": "A-",
                "gpa": "A- (84%)",
                "overall_score": 84.5,
                "data": {
                    "age": 16,
                    "gender": "male",
                    "school_type": "public",
                    "parent_education": "graduate",
                    "study_hours": 4.5,
                    "attendance_percentage": 82.0,
                    "internet_access": "yes",
                    "travel_time": "15-30 min",
                    "extra_activities": "yes",
                    "study_method": "notes",
                    "math_score": 68.0,
                    "science_score": 76.0,
                    "english_score": 82.0
                },
                "homework_completed": 80,
                "homework_total": 120,
                "exams_completed": 5,
                "exams_total": 12
            },
            {
                "id": "sophia",
                "name": "Sophia Martinez",
                "role": "At-Risk Focus",
                "avatar": "assets/avatar_sophia.png",
                "grade": "C",
                "gpa": "C (54%)",
                "overall_score": 54.0,
                "data": {
                    "age": 16,
                    "gender": "female",
                    "school_type": "public",
                    "parent_education": "high school",
                    "study_hours": 1.5,
                    "attendance_percentage": 62.0,
                    "internet_access": "no",
                    "travel_time": "30-60 min",
                    "extra_activities": "no",
                    "study_method": "textbook",
                    "math_score": 42.0,
                    "science_score": 51.0,
                    "english_score": 56.0
                },
                "homework_completed": 45,
                "homework_total": 120,
                "exams_completed": 3,
                "exams_total": 12
            },
            {
                "id": "emily",
                "name": "Emily Watson",
                "role": "Rising Talent",
                "avatar": "assets/avatar_emily.png",
                "grade": "B+",
                "gpa": "B+ (78%)",
                "overall_score": 78.4,
                "data": {
                    "age": 17,
                    "gender": "female",
                    "school_type": "private",
                    "parent_education": "graduate",
                    "study_hours": 3.5,
                    "attendance_percentage": 86.0,
                    "internet_access": "yes",
                    "travel_time": "<15 min",
                    "extra_activities": "yes",
                    "study_method": "online videos",
                    "math_score": 74.0,
                    "science_score": 80.0,
                    "english_score": 81.0
                },
                "homework_completed": 92,
                "homework_total": 120,
                "exams_completed": 8,
                "exams_total": 12
            }
        ]
    }

@app.get("/api/v1/analytics/dataset-insights", tags=["Analytics"])
def get_dataset_insights():
    """Returns aggregated real statistical distributions from the 15,000-record dataset for clean charting."""
    try:
        if os.path.exists(DEFAULT_DATASET_PATH):
            df = pd.read_csv(DEFAULT_DATASET_PATH)
        else:
            raise FileNotFoundError("Dataset file not found")
        
        # 1. Study Hours vs Average Score
        df["study_hour_bin"] = pd.cut(df["study_hours"], bins=[0, 2, 4, 6, 8, 24], labels=["0-2 hrs", "2-4 hrs", "4-6 hrs", "6-8 hrs", "8+ hrs"])
        hours_score = df.groupby("study_hour_bin", observed=False)["overall_score"].mean().round(1).to_dict()
        
        # 2. Attendance vs Average Score
        df["attendance_bin"] = pd.cut(df["attendance_percentage"], bins=[0, 50, 70, 85, 100], labels=["<50%", "50-70%", "70-85%", "85-100%"])
        att_score = df.groupby("attendance_bin", observed=False)["overall_score"].mean().round(1).to_dict()
        
        # 3. Study Method Performance
        method_stats = df.groupby("study_method")["overall_score"].agg(["mean", "count"]).round(1).reset_index().to_dict(orient="records")
        
        # 4. Subject Score Averages
        subject_avgs = {
            "math": round(df["math_score"].mean(), 1),
            "science": round(df["science_score"].mean(), 1),
            "english": round(df["english_score"].mean(), 1),
            "overall": round(df["overall_score"].mean(), 1)
        }
        
        # 5. Grade Tier Distribution
        tier_counts = {
            "A (High: 80+)": int((df["overall_score"] >= 80).sum()),
            "B (Medium: 60-79)": int(((df["overall_score"] >= 60) & (df["overall_score"] < 80)).sum()),
            "C (Low: <60)": int((df["overall_score"] < 60).sum())
        }
        
        return {
            "total_records": len(df),
            "hours_vs_score": hours_score,
            "attendance_vs_score": att_score,
            "method_stats": method_stats,
            "subject_averages": subject_avgs,
            "grade_distribution": tier_counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics calculation error: {str(e)}")

@app.post("/api/v1/predict", response_model=SinglePredictionResponse, tags=["Inference"])
def predict_student(student: StudentInput):
    """
    Predict academic outcome tier for an individual student and provide XAI risk explanation.
    """
    try:
        student_data = student.model_dump()
        result = ModelTrainer.predict_single(student_data)
        
        # Compute explainability & risk attribution
        explanation = ExplainabilityEngine.analyze_student_risk_factors(student_data, result["prediction"])
        
        return SinglePredictionResponse(
            status="success",
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_name=result["model_name"],
            used_features=result["used_features"],
            explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/v1/batch-predict", tags=["Inference"])
def predict_batch_students(batch: BatchStudentInput):
    """
    Perform batch inference on a collection of student records.
    """
    try:
        student_dicts = [s.model_dump() for s in batch.students]
        df_batch = pd.DataFrame(student_dicts)
        results_df = ModelTrainer.predict_batch(df_batch)
        return {
            "status": "success",
            "count": len(results_df),
            "results": results_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/api/v1/train", tags=["Training"])
def train_pipeline(req: TrainRequest):
    """
    Trigger end-to-end retraining with specified feature selection method and target configuration.
    """
    try:
        trainer = ModelTrainer(
            target_scheme=req.target_scheme,
            feature_selection_method=req.feature_selection_method,
            n_features=req.n_features
        )
        report = trainer.run_full_pipeline(
            tune_hyperparameters=req.tune_hyperparameters,
            save_artifacts=True
        )
        return {
            "status": "success",
            "best_model": report["best_model_name"],
            "best_f1_score": report["best_score_f1"],
            "selected_features": report["selected_features"],
            "message": "Model and preprocessors trained and serialized successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failure: {str(e)}")
