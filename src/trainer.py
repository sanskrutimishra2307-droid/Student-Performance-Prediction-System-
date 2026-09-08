"""
Model Trainer and Benchmarking Engine.
Orchestrates preprocessing, feature selection, cross-validation, hyperparameter tuning,
model evaluation, and persistent artifact serialization.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

from src.config import (
    MODELS_DIR,
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURE_SELECTOR_PATH,
    METRICS_REPORT_PATH,
    RANDOM_STATE,
    CV_FOLDS,
    DEFAULT_TARGET_SCHEME
)
from src.data_loader import DataLoader
from src.preprocessing import PreprocessingPipeline
from src.feature_selection import FeatureSelectorSuite
from src.models import ModelFactory
from src.evaluation import ModelEvaluator

class ModelTrainer:
    """
    End-to-end training, benchmarking, and inference manager.
    """
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        target_scheme: str = DEFAULT_TARGET_SCHEME,
        feature_selection_method: str = "consensus",
        n_features: int = 8,
        scaler_type: str = "standard"
    ):
        self.dataset_path = dataset_path
        self.target_scheme = target_scheme
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.scaler_type = scaler_type
        
        self.data_loader = DataLoader(dataset_path=dataset_path, target_scheme=target_scheme)
        self.preprocessor = PreprocessingPipeline(scaler_type=scaler_type)
        self.feature_selector = FeatureSelectorSuite(n_features_to_select=n_features)
        
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.best_score: float = 0.0
        self.selected_feature_names: List[str] = []
        self.benchmark_results: Dict[str, Any] = {}

    def run_full_pipeline(
        self,
        tune_hyperparameters: bool = False,
        save_artifacts: bool = True
    ) -> Dict[str, Any]:
        """
        Executes complete training pipeline:
        1. Ingest & split dataset
        2. Fit preprocessing & feature engineering
        3. Execute feature selection suite
        4. Benchmark 8 classification models
        5. Tune best model if requested
        6. Persist artifacts
        """
        # 1. Load Data
        X_train, X_test, y_train, y_test = self.data_loader.get_train_test_split()
        
        # 2. Preprocess & Encode
        X_train_proc, y_train_enc, all_feature_names = self.preprocessor.fit_transform(X_train, y_train)
        X_test_proc = self.preprocessor.transform(X_test)
        y_test_enc = self.preprocessor.encode_target(y_test)
        
        class_names = list(self.preprocessor.classes_)
        
        # 3. Feature Selection
        if self.feature_selection_method == "all_features":
            self.selected_feature_names = all_feature_names
            feature_indices = list(range(len(all_feature_names)))
            X_train_sub = X_train_proc
            X_test_sub = X_test_proc
        else:
            # Run selector suite
            fs_summary = self.feature_selector.run_all_methods(X_train_proc, y_train_enc, all_feature_names)
            
            # Select target method features
            if self.feature_selection_method in self.feature_selector.selected_features:
                self.selected_feature_names = self.feature_selector.selected_features[self.feature_selection_method]
            else:
                self.selected_feature_names, _ = self.feature_selector.ensemble_consensus(X_train_proc, y_train_enc, all_feature_names)
                
            feature_indices = [all_feature_names.index(f) for f in self.selected_feature_names]
            X_train_sub = X_train_proc[:, feature_indices]
            X_test_sub = X_test_proc[:, feature_indices]
            
        # 4. Model Benchmarking
        models = ModelFactory.get_models()
        results = {}
        best_f1 = -1.0
        best_name = None
        best_clf = None
        
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        for name, clf in models.items():
            # Cross validation on training subset
            cv_scores = cross_val_score(clf, X_train_sub, y_train_enc, cv=cv, scoring="f1_weighted", n_jobs=-1)
            cv_acc = cross_val_score(clf, X_train_sub, y_train_enc, cv=cv, scoring="accuracy", n_jobs=-1)
            
            # Fit on full training set and evaluate on held-out test set
            clf.fit(X_train_sub, y_train_enc)
            y_pred = clf.predict(X_test_sub)
            y_prob = clf.predict_proba(X_test_sub) if hasattr(clf, "predict_proba") else None
            
            eval_metrics = ModelEvaluator.evaluate_predictions(y_test_enc, y_pred, y_prob, class_names=class_names)
            
            results[name] = {
                "cv_f1_mean": float(np.mean(cv_scores)),
                "cv_f1_std": float(np.std(cv_scores)),
                "cv_acc_mean": float(np.mean(cv_acc)),
                "cv_acc_std": float(np.std(cv_acc)),
                "test_metrics": eval_metrics
            }
            
            if eval_metrics["f1_weighted"] > best_f1:
                best_f1 = eval_metrics["f1_weighted"]
                best_name = name
                best_clf = clf
                
        # 5. Optional Hyperparameter Tuning on the Best Model
        if tune_hyperparameters and best_name in ModelFactory.get_hyperparameter_grids():
            grid = ModelFactory.get_hyperparameter_grids()[best_name]
            grid_search = GridSearchCV(
                estimator=best_clf,
                param_grid=grid,
                cv=cv,
                scoring="f1_weighted",
                n_jobs=-1
            )
            grid_search.fit(X_train_sub, y_train_enc)
            best_clf = grid_search.best_estimator_
            
            # Re-evaluate tuned model
            y_pred_tuned = best_clf.predict(X_test_sub)
            y_prob_tuned = best_clf.predict_proba(X_test_sub) if hasattr(best_clf, "predict_proba") else None
            results[best_name]["test_metrics"] = ModelEvaluator.evaluate_predictions(
                y_test_enc, y_pred_tuned, y_prob_tuned, class_names=class_names
            )
            results[best_name]["best_params"] = grid_search.best_params_
            
        self.best_model_name = best_name
        self.best_model = best_clf
        self.best_score = best_f1
        self.benchmark_results = results
        
        # 6. Persist Artifacts
        if save_artifacts:
            self.save_artifacts()
            
        return {
            "best_model_name": self.best_model_name,
            "best_score_f1": self.best_score,
            "selected_features": self.selected_feature_names,
            "benchmark_results": self.benchmark_results,
            "class_names": class_names
        }

    def save_artifacts(self):
        """
        Serializes best model, preprocessor, feature selector, and metrics summary.
        """
        MODELS_DIR.mkdir(exist_ok=True)
        
        metadata = {
            "best_model_name": self.best_model_name,
            "best_f1_score": self.best_score,
            "target_scheme": self.target_scheme,
            "selected_features": self.selected_feature_names,
            "all_features": self.preprocessor.feature_names_out,
            "class_names": list(self.preprocessor.classes_),
            "feature_selection_method": self.feature_selection_method
        }
        
        joblib.dump(self.best_model, BEST_MODEL_PATH)
        joblib.dump(self.preprocessor, PREPROCESSOR_PATH)
        joblib.dump(self.feature_selector, FEATURE_SELECTOR_PATH)
        
        # Save metrics summary as JSON
        serializable_results = {
            "metadata": metadata,
            "models": {
                k: {
                    "cv_f1_mean": v["cv_f1_mean"],
                    "cv_acc_mean": v["cv_acc_mean"],
                    "test_accuracy": v["test_metrics"]["accuracy"],
                    "test_f1_weighted": v["test_metrics"]["f1_weighted"],
                    "test_precision_weighted": v["test_metrics"]["precision_weighted"],
                    "test_recall_weighted": v["test_metrics"]["recall_weighted"],
                    "roc_auc": v["test_metrics"]["roc_auc"]
                }
                for k, v in self.benchmark_results.items()
            }
        }
        
        with open(METRICS_REPORT_PATH, "w") as f:
            json.dump(serializable_results, f, indent=4)
            
        print(f"Artifacts persisted successfully in '{MODELS_DIR}'")

    @classmethod
    def load_trained_system(cls):
        """
        Loads saved model artifacts for immediate inference.
        """
        if not BEST_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError("Model artifacts not found. Please train the system first.")
            
        model = joblib.load(BEST_MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        feature_selector = joblib.load(FEATURE_SELECTOR_PATH) if FEATURE_SELECTOR_PATH.exists() else None
        
        metadata = {}
        if METRICS_REPORT_PATH.exists():
            with open(METRICS_REPORT_PATH, "r") as f:
                metadata = json.load(f).get("metadata", {})
                
        return model, preprocessor, feature_selector, metadata

    @classmethod
    def predict_single(cls, student_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single student inference with probability distribution and confidence score.
        """
        model, preprocessor, _, metadata = cls.load_trained_system()
        selected_features = metadata.get("selected_features", [])
        all_features = metadata.get("all_features", [])
        
        df_input = pd.DataFrame([student_dict])
        X_proc = preprocessor.transform(df_input)
        
        if selected_features and all_features:
            indices = [all_features.index(f) for f in selected_features if f in all_features]
            X_sub = X_proc[:, indices]
        else:
            X_sub = X_proc
            
        pred_enc = model.predict(X_sub)[0]
        pred_label = preprocessor.decode_target(np.array([pred_enc]))[0]
        
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_sub)[0]
            for cls_name, prob in zip(preprocessor.classes_, probs):
                probabilities[cls_name] = float(prob)
                
        confidence = float(np.max(list(probabilities.values()))) if probabilities else 1.0
        
        return {
            "prediction": str(pred_label),
            "confidence": confidence,
            "probabilities": probabilities,
            "model_name": metadata.get("best_model_name", "Best Model"),
            "used_features": selected_features
        }

    @classmethod
    def predict_batch(cls, df_batch: pd.DataFrame) -> pd.DataFrame:
        """
        Batch inference on student dataframe.
        """
        model, preprocessor, _, metadata = cls.load_trained_system()
        selected_features = metadata.get("selected_features", [])
        all_features = metadata.get("all_features", [])
        
        df_clean = df_batch.copy()
        X_proc = preprocessor.transform(df_clean)
        
        if selected_features and all_features:
            indices = [all_features.index(f) for f in selected_features if f in all_features]
            X_sub = X_proc[:, indices]
        else:
            X_sub = X_proc
            
        preds_enc = model.predict(X_sub)
        preds_label = preprocessor.decode_target(preds_enc)
        
        results_df = df_batch.copy()
        results_df["predicted_performance"] = preds_label
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_sub)
            results_df["prediction_confidence"] = np.max(probs, axis=1).round(3)
            for i, cls_name in enumerate(preprocessor.classes_):
                results_df[f"prob_{cls_name.lower().replace(' ', '_')}"] = probs[:, i].round(3)
                
        return results_df
