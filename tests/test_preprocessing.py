"""
Test Suite for Data Loader and Preprocessing Pipeline.
"""

import pytest
import pandas as pd
import numpy as np

from src.data_loader import DataLoader
from src.preprocessing import PreprocessingPipeline, StudentFeatureEngineer
from src.config import DEFAULT_DATASET_PATH

def test_data_loader_initialization():
    loader = DataLoader(str(DEFAULT_DATASET_PATH))
    df = loader.load_data()
    assert df is not None
    assert len(df) > 0
    assert "study_hours" in df.columns
    assert "attendance_percentage" in df.columns

def test_train_test_split():
    loader = DataLoader(str(DEFAULT_DATASET_PATH), target_scheme="3-tier")
    X_train, X_test, y_train, y_test = loader.get_train_test_split(test_size=0.2)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert set(y_train.unique()).issubset({"Distinction", "Pass", "At-Risk"})

def test_feature_engineering():
    engineer = StudentFeatureEngineer(create_interactions=True)
    df_sample = pd.DataFrame({
        "study_hours": [4.0, 1.5],
        "attendance_percentage": [90.0, 55.0],
        "math_score": [70.0, 40.0],
        "science_score": [80.0, 42.0],
        "english_score": [75.0, 45.0]
    })
    df_transformed = engineer.transform(df_sample)
    assert "academic_dedication" in df_transformed.columns
    assert "low_attendance_risk" in df_transformed.columns
    assert "stem_avg" in df_transformed.columns
    assert df_transformed["low_attendance_risk"].iloc[1] == 1
    assert df_transformed["low_attendance_risk"].iloc[0] == 0

def test_preprocessing_pipeline():
    loader = DataLoader(str(DEFAULT_DATASET_PATH))
    X, y = loader.prepare_features_and_target()
    
    pipeline = PreprocessingPipeline(scaler_type="standard")
    X_proc, y_enc, feat_names = pipeline.fit_transform(X.head(50), y.head(50))
    
    assert X_proc.shape[0] == 50
    assert len(feat_names) == X_proc.shape[1]
    assert len(y_enc) == 50
    
    # Test new transform
    X_new_proc = pipeline.transform(X.iloc[50:60])
    assert X_new_proc.shape[0] == 10
    assert X_new_proc.shape[1] == X_proc.shape[1]
