"""
Data Ingestion, Validation, and Splitting Module.
Handles CSV ingestion, data cleaning, target transformation, and stratified train-test splitting.
"""

import os
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    DEFAULT_DATASET_PATH,
    ID_COLUMN,
    RAW_TARGET_COLUMN,
    RAW_SCORE_COLUMN,
    TARGET_SCHEMES,
    DEFAULT_TARGET_SCHEME,
    RANDOM_STATE,
    TEST_SIZE,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

class DataLoader:
    """
    Robust data loader and validator for Student Performance records.
    """
    def __init__(self, dataset_path: Optional[str] = None, target_scheme: str = DEFAULT_TARGET_SCHEME):
        self.dataset_path = dataset_path or str(DEFAULT_DATASET_PATH)
        self.target_scheme = target_scheme
        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads dataset from CSV file or creates benchmark if missing.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at '{self.dataset_path}'. Please provide a valid CSV file.")
            
        df = pd.read_csv(self.dataset_path)
        self.raw_df = df.copy()
        
        # Drop duplicates if any
        df = df.drop_duplicates().reset_index(drop=True)
        
        # Standardize column names (lowercase & stripped)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Validate required target column
        if RAW_TARGET_COLUMN not in df.columns and RAW_SCORE_COLUMN not in df.columns:
            raise ValueError(f"Dataset must contain either '{RAW_TARGET_COLUMN}' or '{RAW_SCORE_COLUMN}' column.")
            
        # Clean target column strings
        if RAW_TARGET_COLUMN in df.columns:
            df[RAW_TARGET_COLUMN] = df[RAW_TARGET_COLUMN].astype(str).str.strip().str.lower()
            
        self.clean_df = df
        return self.clean_df

    def prepare_features_and_target(
        self,
        df: Optional[pd.DataFrame] = None,
        include_subject_scores: bool = True,
        target_scheme: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extracts feature matrix X and target vector y with appropriate mapping.
        """
        if df is None:
            if self.clean_df is None:
                self.load_data()
            df = self.clean_df.copy()
        else:
            df = df.copy()
            
        scheme_key = target_scheme or self.target_scheme
        scheme_cfg = TARGET_SCHEMES.get(scheme_key, TARGET_SCHEMES[DEFAULT_TARGET_SCHEME])
        mapping = scheme_cfg["mapping"]
        
        # Derive target if needed
        if RAW_TARGET_COLUMN in df.columns:
            y = df[RAW_TARGET_COLUMN].map(mapping)
            # Handle unmapped values if any
            if y.isnull().any():
                # Fallback to direct value or default
                y = y.fillna("Pass")
        elif RAW_SCORE_COLUMN in df.columns:
            # Derive grade from overall_score
            def score_to_tier(score):
                if scheme_key == "binary":
                    return "Pass" if score >= 50.0 else "Fail"
                elif scheme_key == "3-tier":
                    if score >= 75.0:
                        return "Distinction"
                    elif score >= 50.0:
                        return "Pass"
                    else:
                        return "At-Risk"
                else:
                    if score >= 85: return "A"
                    elif score >= 75: return "B"
                    elif score >= 65: return "C"
                    elif score >= 55: return "D"
                    elif score >= 45: return "E"
                    else: return "F"
            y = df[RAW_SCORE_COLUMN].apply(score_to_tier)
        else:
            raise ValueError("No target column identifiable in DataFrame.")
            
        # Drop ID and target columns from feature matrix X
        drop_cols = [ID_COLUMN, RAW_TARGET_COLUMN, RAW_SCORE_COLUMN, "target"]
        if not include_subject_scores:
            drop_cols.extend(["math_score", "science_score", "english_score"])
            
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        return X, y

    def get_train_test_split(
        self,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        include_subject_scores: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits dataset into stratified training and testing sets.
        """
        X, y = self.prepare_features_and_target(include_subject_scores=include_subject_scores)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        return X_train, X_test, y_train, y_test

    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Returns rich statistical and structural overview of the dataset.
        """
        if self.clean_df is None:
            self.load_data()
            
        df = self.clean_df
        return {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "numerical_columns": [c for c in df.select_dtypes(include=[np.number]).columns if c != ID_COLUMN],
            "categorical_columns": [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in [RAW_TARGET_COLUMN]],
            "missing_values": df.isnull().sum().to_dict(),
            "target_distribution": df[RAW_TARGET_COLUMN].value_counts().to_dict() if RAW_TARGET_COLUMN in df.columns else {},
            "target_scheme": self.target_scheme
        }
