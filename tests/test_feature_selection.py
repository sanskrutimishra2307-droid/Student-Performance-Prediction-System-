"""
Test Suite for Feature Selection Suite.
"""

import pytest
import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.preprocessing import PreprocessingPipeline
from src.feature_selection import FeatureSelectorSuite
from src.config import DEFAULT_DATASET_PATH

@pytest.fixture
def sample_data():
    loader = DataLoader(str(DEFAULT_DATASET_PATH))
    X, y = loader.prepare_features_and_target()
    pipeline = PreprocessingPipeline()
    X_proc, y_enc, feat_names = pipeline.fit_transform(X.head(200), y.head(200))
    return X_proc, y_enc, feat_names

def test_mutual_info(sample_data):
    X, y, feat_names = sample_data
    suite = FeatureSelectorSuite(n_features_to_select=5)
    top_k, scores = suite.filter_mutual_info(X, y, feat_names)
    assert len(top_k) == 5
    assert len(scores) == len(feat_names)

def test_anova_f(sample_data):
    X, y, feat_names = sample_data
    suite = FeatureSelectorSuite(n_features_to_select=5)
    top_k, scores = suite.filter_anova_f(X, y, feat_names)
    assert len(top_k) == 5
    assert len(scores) == len(feat_names)

def test_lasso_l1(sample_data):
    X, y, feat_names = sample_data
    suite = FeatureSelectorSuite(n_features_to_select=5)
    top_k, scores = suite.embedded_lasso_l1(X, y, feat_names)
    assert len(top_k) == 5
    assert len(scores) == len(feat_names)

def test_consensus_ensemble(sample_data):
    X, y, feat_names = sample_data
    suite = FeatureSelectorSuite(n_features_to_select=5)
    top_k, scores = suite.ensemble_consensus(X, y, feat_names)
    assert len(top_k) == 5
    assert len(scores) == len(feat_names)

def test_dimensionality_curve(sample_data):
    X, y, feat_names = sample_data
    suite = FeatureSelectorSuite(n_features_to_select=5)
    curve = suite.evaluate_dimensionality_curve(X, y, feat_names, method="consensus", step=3)
    assert "k_values" in curve
    assert "f1_scores" in curve
    assert len(curve["k_values"]) == len(curve["f1_scores"])
    assert curve["optimal_k"] > 0
