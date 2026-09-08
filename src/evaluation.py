"""
Comprehensive Model Evaluation & Metrics Suite.
Computes multi-class and binary metrics, confusion matrices, ROC curves, and radar visual data.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

class ModelEvaluator:
    """
    Evaluator for classification models on student performance prediction.
    """
    
    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Computes all standard performance metrics for a model.
        """
        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        kappa = float(cohen_kappa_score(y_true, y_pred))
        
        # ROC AUC
        roc_auc = None
        if y_prob is not None:
            try:
                if y_prob.shape[1] == 2:
                    roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
            except Exception:
                roc_auc = None
                
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
        
        # Classification report dictionary
        target_names = class_names if class_names else [str(c) for c in np.unique(y_true)]
        clf_rep = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        
        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision_macro": prec_macro,
            "precision_weighted": prec_weighted,
            "recall_macro": rec_macro,
            "recall_weighted": rec_weighted,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "cohen_kappa": kappa,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_norm.tolist(),
            "classification_report": clf_rep,
            "class_names": target_names
        }

    @staticmethod
    def get_roc_curve_data(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, Any]:
        """
        Generates ROC curve points (FPR, TPR, AUC) for each class (One-vs-Rest).
        """
        roc_data = {}
        n_classes = len(class_names)
        
        # Binarize labels if multi-class
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            auc_val = roc_auc_score(y_true, y_prob[:, 1])
            roc_data[class_names[1]] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(auc_val)
            }
        else:
            for i, class_label in enumerate(class_names):
                y_binary = (y_true == i).astype(int)
                if np.sum(y_binary) > 0 and len(np.unique(y_binary)) > 1:
                    fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
                    auc_val = roc_auc_score(y_binary, y_prob[:, i])
                    roc_data[class_label] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "auc": float(auc_val)
                    }
        return roc_data
