"""
Feature Selection Suite for Student Performance Prediction.
Implements Filter, Wrapper, Embedded, and Ensemble Consensus Feature Selection Algorithms.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    f_classif,
    chi2,
    RFECV,
    SequentialFeatureSelector,
    SelectFromModel,
    VarianceThreshold
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

from src.config import RANDOM_STATE, CV_FOLDS

class FeatureSelectorSuite:
    """
    Comprehensive feature selection suite providing Filter, Wrapper, Embedded,
    and Consensus ranking engines.
    """
    def __init__(
        self,
        n_features_to_select: int = 8,
        cv_folds: int = CV_FOLDS,
        random_state: int = RANDOM_STATE
    ):
        self.k = n_features_to_select
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.rankings: Dict[str, Dict[str, float]] = {}
        self.selected_features: Dict[str, List[str]] = {}

    def filter_mutual_info(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[List[str], Dict[str, float]]:
        """
        Filter Method: Mutual Information Classification.
        Measures the non-linear dependency between variables and the target.
        """
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        score_dict = {f: float(s) for f, s in zip(feature_names, mi_scores)}
        sorted_features = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        top_k = [f[0] for f in sorted_features[:self.k]]
        
        self.rankings["mutual_info"] = score_dict
        self.selected_features["mutual_info"] = top_k
        return top_k, score_dict

    def filter_anova_f(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[List[str], Dict[str, float]]:
        """
        Filter Method: ANOVA F-value (f_classif).
        Computes the linear dependency and variance ratio between classes.
        """
        f_scores, p_values = f_classif(X, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        score_dict = {f: float(s) for f, s in zip(feature_names, f_scores)}
        sorted_features = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        top_k = [f[0] for f in sorted_features[:self.k]]
        
        self.rankings["anova_f"] = score_dict
        self.selected_features["anova_f"] = top_k
        return top_k, score_dict

    def filter_chi2(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[List[str], Dict[str, float]]:
        """
        Filter Method: Chi-Square Test.
        Requires non-negative features.
        """
        # Scale to [0, 1] for non-negative Chi2 computation
        scaler = MinMaxScaler()
        X_pos = scaler.fit_transform(X)
        
        chi2_scores, _ = chi2(X_pos, y)
        chi2_scores = np.nan_to_num(chi2_scores, nan=0.0)
        score_dict = {f: float(s) for f, s in zip(feature_names, chi2_scores)}
        sorted_features = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        top_k = [f[0] for f in sorted_features[:self.k]]
        
        self.rankings["chi2"] = score_dict
        self.selected_features["chi2"] = top_k
        return top_k, score_dict

    def wrapper_rfecv(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[List[str], Dict[str, float]]:
        """
        Wrapper Method: Recursive Feature Elimination with Cross-Validation (RFECV).
        Prunes least important features recursively using a Random Forest estimator.
        """
        estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        cv = StratifiedKFold(n_splits=min(3, self.cv_folds), shuffle=True, random_state=self.random_state)
        
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring="f1_weighted",
            min_features_to_select=max(2, min(self.k, len(feature_names)))
        )
        selector.fit(X, y)
        
        # Ranking: 1 is best, higher is worse. We invert ranking to score
        ranks = selector.ranking_
        max_rank = float(np.max(ranks))
        score_dict = {f: float(max_rank - r + 1.0) for f, r in zip(feature_names, ranks)}
        
        sorted_features = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        top_k = [f[0] for f in sorted_features[:self.k]]
        
        self.rankings["rfecv"] = score_dict
        self.selected_features["rfecv"] = top_k
        return top_k, score_dict

    def wrapper_sfs(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], direction: str = "forward") -> Tuple[List[str], Dict[str, float]]:
        """
        Wrapper Method: Sequential Feature Selector (SFS - Forward or Backward).
        Greedily builds or trims the optimal subset based on cross-validation performance.
        """
        estimator = LogisticRegression(max_iter=500, random_state=self.random_state)
        cv = StratifiedKFold(n_splits=min(3, self.cv_folds), shuffle=True, random_state=self.random_state)
        
        n_features = min(self.k, len(feature_names))
        sfs = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=n_features,
            direction=direction,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1
        )
        sfs.fit(X, y)
        
        mask = sfs.get_support()
        top_k = [f for f, m in zip(feature_names, mask) if m]
        score_dict = {f: 1.0 if m else 0.0 for f, m in zip(feature_names, mask)}
        
        method_name = f"sfs_{direction}"
        self.rankings[method_name] = score_dict
        self.selected_features[method_name] = top_k
        return top_k, score_dict

    def embedded_lasso_l1(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[List[str], Dict[str, float]]:
        """
        Embedded Method: L1 Regularized Logistic Regression (Lasso).
        Drives non-informative feature weights strictly to zero.
        """
        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=0.1,
            max_iter=1000,
            random_state=self.random_state
        )
        clf.fit(X, y)
        
        # Absolute coefficient magnitude across all classes
        coef_mag = np.mean(np.abs(clf.coef_), axis=0) if clf.coef_.ndim > 1 else np.abs(clf.coef_[0])
        score_dict = {f: float(s) for f, s in zip(feature_names, coef_mag)}
        sorted_features = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        top_k = [f[0] for f in sorted_features[:self.k]]
        
        self.rankings["lasso_l1"] = score_dict
        self.selected_features["lasso_l1"] = top_k
        return top_k, score_dict

    def embedded_random_forest(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[List[str], Dict[str, float]]:
        """
        Embedded Method: Random Forest Mean Decrease in Impurity (Gini MDI).
        """
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        score_dict = {f: float(s) for f, s in zip(feature_names, importances)}
        sorted_features = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        top_k = [f[0] for f in sorted_features[:self.k]]
        
        self.rankings["random_forest_mdi"] = score_dict
        self.selected_features["random_forest_mdi"] = top_k
        return top_k, score_dict

    def embedded_xgboost(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[List[str], Dict[str, float]]:
        """
        Embedded Method: XGBoost Feature Importance (Gain).
        """
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=self.random_state,
            eval_metric="mlogloss",
            n_jobs=-1
        )
        xgb.fit(X, y)
        
        importances = xgb.feature_importances_
        score_dict = {f: float(s) for f, s in zip(feature_names, importances)}
        sorted_features = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        top_k = [f[0] for f in sorted_features[:self.k]]
        
        self.rankings["xgboost_mdi"] = score_dict
        self.selected_features["xgboost_mdi"] = top_k
        return top_k, score_dict

    def ensemble_consensus(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[List[str], Dict[str, float]]:
        """
        Consensus Feature Selection Engine.
        Aggregates normalized rank percentiles from Filter (MI, ANOVA), Wrapper (RFECV),
        and Embedded (Random Forest, XGBoost, Lasso) methods into a robust ensemble rank.
        """
        # Run all individual methods if not already run
        if "mutual_info" not in self.rankings:
            self.filter_mutual_info(X, y, feature_names)
        if "anova_f" not in self.rankings:
            self.filter_anova_f(X, y, feature_names)
        if "random_forest_mdi" not in self.rankings:
            self.embedded_random_forest(X, y, feature_names)
        if "xgboost_mdi" not in self.rankings:
            self.embedded_xgboost(X, y, feature_names)
        if "lasso_l1" not in self.rankings:
            self.embedded_lasso_l1(X, y, feature_names)
            
        # Normalize each method's scores to [0, 1] range
        normalized_scores: Dict[str, Dict[str, float]] = {}
        for method, scores in self.rankings.items():
            if method.startswith("consensus"):
                continue
            max_val = max(scores.values()) if scores.values() else 1.0
            min_val = min(scores.values()) if scores.values() else 0.0
            rng = max_val - min_val if (max_val - min_val) > 0 else 1.0
            normalized_scores[method] = {f: (s - min_val) / rng for f, s in scores.items()}
            
        # Calculate ensemble average score for each feature
        consensus_scores: Dict[str, float] = {}
        for f in feature_names:
            feature_vals = [normalized_scores[m].get(f, 0.0) for m in normalized_scores]
            consensus_scores[f] = float(np.mean(feature_vals))
            
        sorted_features = sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = [f[0] for f in sorted_features[:self.k]]
        
        self.rankings["consensus"] = consensus_scores
        self.selected_features["consensus"] = top_k
        return top_k, consensus_scores

    def run_all_methods(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Executes all feature selection algorithms and returns comparative summary.
        """
        self.filter_mutual_info(X, y, feature_names)
        self.filter_anova_f(X, y, feature_names)
        self.filter_chi2(X, y, feature_names)
        self.embedded_random_forest(X, y, feature_names)
        self.embedded_xgboost(X, y, feature_names)
        self.embedded_lasso_l1(X, y, feature_names)
        self.ensemble_consensus(X, y, feature_names)
        
        summary = {}
        for method in self.rankings:
            summary[method] = {
                "top_features": self.selected_features.get(method, []),
                "scores": self.rankings[method]
            }
        return summary

    def evaluate_dimensionality_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        method: str = "consensus",
        step: int = 1
    ) -> Dict[str, Any]:
        """
        Computes the classification performance (Accuracy & F1) as feature count k scales from 1 to N.
        Demonstrates the efficiency of feature selection vs full dimensionality.
        """
        if method not in self.rankings:
            if method == "consensus":
                self.ensemble_consensus(X, y, feature_names)
            else:
                getattr(self, f"filter_{method}", getattr(self, f"embedded_{method}"))(X, y, feature_names)
                
        sorted_ranking = sorted(self.rankings[method].items(), key=lambda x: x[1], reverse=True)
        ordered_features = [x[0] for x in sorted_ranking]
        
        k_values = list(range(1, len(ordered_features) + 1, step))
        if len(ordered_features) not in k_values:
            k_values.append(len(ordered_features))
            
        accuracies = []
        f1_scores = []
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        evaluator = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        
        for k in k_values:
            top_k_names = ordered_features[:k]
            indices = [feature_names.index(name) for name in top_k_names]
            X_subset = X[:, indices]
            
            acc = cross_val_score(evaluator, X_subset, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
            f1 = cross_val_score(evaluator, X_subset, y, cv=cv, scoring="f1_weighted", n_jobs=-1).mean()
            
            accuracies.append(float(acc))
            f1_scores.append(float(f1))
            
        return {
            "method": method,
            "k_values": k_values,
            "accuracies": accuracies,
            "f1_scores": f1_scores,
            "optimal_k": int(k_values[np.argmax(f1_scores)]),
            "max_f1": float(np.max(f1_scores))
        }
