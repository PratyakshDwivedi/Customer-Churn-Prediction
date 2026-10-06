"""Tests for training, evaluation and model selection."""

from __future__ import annotations

import pandas as pd

from src.config import MODEL_FEATURES, TARGET_COLUMN
from src.data.preprocessing import clean_data
from src.features.engineering import engineer_features
from src.models.evaluate import evaluate_all
from src.models.selection import select_best_model
from src.models.train import (
    fit_preprocessor,
    split_features_target,
    train_candidate_models,
)


def _prepare(raw_df):
    engineered = engineer_features(clean_data(raw_df))
    return engineered


def test_split_is_stratified(raw_df, tmp_config):
    data = split_features_target(_prepare(raw_df), app_config=tmp_config)
    # Train/test churn rates should be close under stratification.
    assert abs(data.y_train.mean() - data.y_test.mean()) < 0.03
    assert set(data.X_train.columns) == set(MODEL_FEATURES)


def test_trains_both_models(raw_df, tmp_config):
    data = split_features_target(_prepare(raw_df), app_config=tmp_config)
    pre = fit_preprocessor(data.X_train)
    models = train_candidate_models(
        pre, data.X_train, data.y_train, tmp_config.random_seed
    )
    assert set(models.keys()) == {"logistic_regression", "random_forest"}


def test_evaluation_metrics_in_valid_range(raw_df, tmp_config):
    data = split_features_target(_prepare(raw_df), app_config=tmp_config)
    pre = fit_preprocessor(data.X_train)
    models = train_candidate_models(
        pre, data.X_train, data.y_train, tmp_config.random_seed
    )
    all_metrics = evaluate_all(models, pre, data.X_test, data.y_test)
    for metrics in all_metrics.values():
        for key in ("f1", "roc_auc", "accuracy", "precision", "recall"):
            assert 0.0 <= metrics[key] <= 1.0
        # Confusion matrix is 2x2.
        assert len(metrics["confusion_matrix"]) == 2
        # ROC-AUC should beat random on a dataset with real signal.
        assert metrics["roc_auc"] > 0.6


def test_selection_picks_highest_f1():
    metrics = {
        "model_a": {"f1": 0.55, "roc_auc": 0.80},
        "model_b": {"f1": 0.62, "roc_auc": 0.78},
    }
    best, reason = select_best_model(metrics)
    assert best == "model_b"
    assert "F1" in reason


def test_selection_tiebreak_on_roc_auc():
    # F1 within tolerance -> ROC-AUC decides.
    metrics = {
        "model_a": {"f1": 0.600, "roc_auc": 0.85},
        "model_b": {"f1": 0.605, "roc_auc": 0.79},
    }
    best, reason = select_best_model(metrics)
    assert best == "model_a"
    assert "ROC-AUC" in reason
