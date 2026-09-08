"""
Feature Engineering, Transformation, and Preprocessing Pipeline.
Implements scikit-learn compatible custom transformers, categorical encodings,
feature scalers, and domain-specific feature synthesis.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ORDINAL_MAPPINGS
)

class StudentFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering transformer for educational analytics.
    Synthesizes academic interaction, efficiency, and risk indicators.
    """
    def __init__(self, create_interactions: bool = True):
        self.create_interactions = create_interactions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create deep copy
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X).copy()
            
        if not self.create_interactions:
            return df
            
        # 1. Study Hours vs Attendance interaction
        if "study_hours" in df.columns and "attendance_percentage" in df.columns:
            # Academic dedication index
            df["academic_dedication"] = (df["study_hours"] * (df["attendance_percentage"] / 100.0)).round(2)
            
        # 2. Low attendance risk indicator (<65%)
        if "attendance_percentage" in df.columns:
            df["low_attendance_risk"] = (df["attendance_percentage"] < 65.0).astype(int)
            
        # 3. High study effort indicator (>6.0 hours)
        if "study_hours" in df.columns:
            df["high_study_effort"] = (df["study_hours"] >= 6.0).astype(int)
            
        # 4. STEM vs Humanities balance (if subject scores present)
        if "math_score" in df.columns and "science_score" in df.columns:
            df["stem_avg"] = ((df["math_score"] + df["science_score"]) / 2.0).round(2)
            
        if "math_score" in df.columns and "science_score" in df.columns and "english_score" in df.columns:
            # Subject score dispersion (consistency across disciplines)
            subj_df = df[["math_score", "science_score", "english_score"]]
            df["score_std"] = subj_df.std(axis=1).round(2)
            df["min_score"] = subj_df.min(axis=1).round(2)
            df["max_score"] = subj_df.max(axis=1).round(2)
            
        return df

class PreprocessingPipeline:
    """
    Configurable scikit-learn preprocessing pipeline orchestrator.
    """
    def __init__(
        self,
        scaler_type: str = "standard",
        impute_strategy: str = "median",
        include_engineering: bool = True
    ):
        self.scaler_type = scaler_type
        self.impute_strategy = impute_strategy
        self.include_engineering = include_engineering
        
        self.feature_engineer = StudentFeatureEngineer(create_interactions=include_engineering)
        self.column_transformer: Optional[ColumnTransformer] = None
        self.label_encoder = LabelEncoder()
        self.feature_names_out: List[str] = []

    def _build_scaler(self):
        if self.scaler_type == "robust":
            return RobustScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        else:
            return StandardScaler()

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Fits transformer on X (and y if provided) and returns processed matrix, encoded y, and feature names.
        """
        # Step 1: Feature Engineering
        X_eng = self.feature_engineer.fit_transform(X)
        
        # Step 2: Separate numeric and categorical column names after engineering
        numeric_cols = [c for c in X_eng.select_dtypes(include=[np.number]).columns]
        cat_cols = [c for c in X_eng.select_dtypes(include=["object", "category"]).columns]
        
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=self.impute_strategy)),
            ("scaler", self._build_scaler())
        ])
        
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        transformers = []
        if numeric_cols:
            transformers.append(("num", num_pipeline, numeric_cols))
        if cat_cols:
            transformers.append(("cat", cat_pipeline, cat_cols))
            
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )
        
        X_processed = self.column_transformer.fit_transform(X_eng)
        
        # Extract feature names
        feature_names = []
        if numeric_cols:
            feature_names.extend(numeric_cols)
        if cat_cols:
            cat_encoder = self.column_transformer.named_transformers_["cat"].named_steps["onehot"]
            cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
            feature_names.extend(list(cat_feature_names))
            
        self.feature_names_out = feature_names
        self.expected_eng_cols = list(X_eng.columns)
        
        # Encode y if provided
        y_encoded = None
        if y is not None:
            y_encoded = self.label_encoder.fit_transform(y)
            
        return X_processed, y_encoded, self.feature_names_out

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms new input data using the fitted pipeline.
        Gracefully handles missing columns by imputing missing values.
        """
        if self.column_transformer is None:
            raise ValueError("PreprocessingPipeline is not fitted yet. Call fit_transform first.")
            
        if isinstance(X, pd.DataFrame):
            df_in = X.copy()
        else:
            df_in = pd.DataFrame(X).copy()
            
        # Feature Engineering
        X_eng = self.feature_engineer.transform(df_in)
        
        # Ensure all columns expected by ColumnTransformer exist
        if hasattr(self, "expected_eng_cols"):
            for col in self.expected_eng_cols:
                if col not in X_eng.columns:
                    X_eng[col] = np.nan
                    
        return self.column_transformer.transform(X_eng)

    def encode_target(self, y: pd.Series) -> np.ndarray:
        return self.label_encoder.transform(y)

    def decode_target(self, y_encoded: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y_encoded)

    @property
    def classes_(self) -> np.ndarray:
        return self.label_encoder.classes_

