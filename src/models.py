"""
Classification Models Factory and Hyperparameter Definitions.
Provides 8 state-of-the-art and baseline machine learning classifiers with tuned parameter grids.
"""

from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE

class ModelFactory:
    """
    Factory for instantiating classification models and their hyperparameter tuning grids.
    """
    
    @staticmethod
    def get_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
        """
        Returns a dictionary of initialized classification algorithms.
        """
        base_lr = LogisticRegression(max_iter=1000, random_state=random_state)
        base_rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        base_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=random_state, eval_metric="mlogloss", n_jobs=-1)
        base_svm = SVC(probability=True, kernel="rbf", C=1.0, random_state=random_state)
        base_knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
        base_dt = DecisionTreeClassifier(max_depth=6, random_state=random_state)
        base_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state, early_stopping=True)
        
        # Stacking Ensemble meta-learner
        stacking_ensemble = StackingClassifier(
            estimators=[
                ("rf", base_rf),
                ("xgb", base_xgb),
                ("svm", base_svm),
                ("knn", base_knn)
            ],
            final_estimator=LogisticRegression(max_iter=500, random_state=random_state),
            n_jobs=-1
        )
        
        # Soft Voting Ensemble
        voting_ensemble = VotingClassifier(
            estimators=[
                ("rf", base_rf),
                ("xgb", base_xgb),
                ("lr", base_lr)
            ],
            voting="soft",
            n_jobs=-1
        )
        
        return {
            "logistic_regression": base_lr,
            "decision_tree": base_dt,
            "random_forest": base_rf,
            "support_vector_machine": base_svm,
            "k_nearest_neighbors": base_knn,
            "xgboost": base_xgb,
            "multi_layer_perceptron": base_mlp,
            "voting_ensemble": voting_ensemble,
            "stacking_ensemble": stacking_ensemble
        }

    @staticmethod
    def get_hyperparameter_grids() -> Dict[str, Dict[str, list]]:
        """
        Returns hyperparameter grids for systematic tuning.
        """
        return {
            "logistic_regression": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs", "saga"]
            },
            "decision_tree": {
                "max_depth": [4, 6, 8, 12, None],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"]
            },
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [6, 10, 15, None],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt", "log2"]
            },
            "support_vector_machine": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]
            },
            "k_nearest_neighbors": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            },
            "xgboost": {
                "n_estimators": [50, 100, 150],
                "max_depth": [3, 4, 6],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0]
            },
            "multi_layer_perceptron": {
                "hidden_layer_sizes": [(64, 32), (100,), (128, 64)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01]
            }
        }
