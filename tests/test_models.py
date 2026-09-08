"""
Test Suite for Model Training, Evaluation, and Inference.
"""

import pytest
import os
import pandas as pd
import numpy as np

from src.trainer import ModelTrainer
from src.config import DEFAULT_DATASET_PATH, SAMPLE_BATCH_PATH

def test_trainer_end_to_end():
    trainer = ModelTrainer(
        dataset_path=str(DEFAULT_DATASET_PATH),
        target_scheme="3-tier",
        feature_selection_method="consensus",
        n_features=6
    )
    report = trainer.run_full_pipeline(tune_hyperparameters=False, save_artifacts=True)
    
    assert report["best_model_name"] is not None
    assert report["best_score_f1"] > 0.60
    assert len(report["selected_features"]) == 6
    assert len(report["class_names"]) == 3

def test_predict_single():
    sample_student = {
        "age": 16,
        "gender": "female",
        "school_type": "public",
        "parent_education": "graduate",
        "study_hours": 6.5,
        "attendance_percentage": 92.0,
        "internet_access": "yes",
        "travel_time": "<15 min",
        "extra_activities": "yes",
        "study_method": "notes",
        "math_score": 85.0,
        "science_score": 88.0,
        "english_score": 90.0
    }
    result = ModelTrainer.predict_single(sample_student)
    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert result["prediction"] in ["Distinction", "Pass", "At-Risk"]

def test_predict_batch():
    df_sample = pd.read_csv(str(SAMPLE_BATCH_PATH))
    results_df = ModelTrainer.predict_batch(df_sample.head(10))
    assert "predicted_performance" in results_df.columns
    assert len(results_df) == 10
