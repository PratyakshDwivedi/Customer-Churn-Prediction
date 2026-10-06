"""Tests for cleaning, feature engineering and the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    AGE_MAX,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.data.preprocessing import build_preprocessor, clean_data
from src.features.engineering import engineer_features


def test_clean_removes_duplicate_ids(raw_df):
    cleaned = clean_data(raw_df)
    assert cleaned["customer_id"].duplicated().sum() == 0


def test_clean_coerces_total_charges_to_numeric(raw_df):
    cleaned = clean_data(raw_df)
    assert pd.api.types.is_numeric_dtype(cleaned["total_charges"])


def test_clean_nulls_impossible_ages(raw_df):
    cleaned = clean_data(raw_df)
    valid_ages = cleaned["age"].dropna()
    assert (valid_ages <= AGE_MAX).all()
    assert (valid_ages >= 18).all()


def test_clean_normalises_categories(raw_df):
    cleaned = clean_data(raw_df)
    # After cleaning, gender should only contain canonical values.
    assert set(cleaned["gender"].dropna().unique()).issubset({"Male", "Female"})
    assert set(cleaned["contract_type"].dropna().unique()).issubset(
        {"Month-to-month", "One year", "Two year"}
    )


def test_clean_encodes_target_binary(raw_df):
    cleaned = clean_data(raw_df)
    assert set(cleaned[TARGET_COLUMN].dropna().unique()).issubset({0, 1})


def test_feature_engineering_adds_expected_columns(raw_df):
    engineered = engineer_features(clean_data(raw_df))
    for col in ("avg_monthly_spend", "charges_per_tenure", "service_count", "tenure_group"):
        assert col in engineered.columns
    # No divide-by-zero infinities leaked in. (NaN is allowed where
    # total_charges was missing - the fitted imputer fills it later.)
    assert not np.isinf(engineered["charges_per_tenure"].to_numpy()).any()
    assert not np.isinf(engineered["avg_monthly_spend"].to_numpy()).any()


def test_preprocessor_fit_transform_shape(raw_df):
    engineered = engineer_features(clean_data(raw_df))
    X = engineered[MODEL_FEATURES]
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    # Rows preserved; columns = numeric + one-hot expanded categoricals.
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] >= len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    # No NaNs survive imputation.
    assert not np.isnan(transformed).any()
