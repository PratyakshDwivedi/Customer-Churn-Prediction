"""Data cleaning and the fitted preprocessing pipeline.

Two distinct responsibilities live here:

1. :func:`clean_data` - a *stateless* cleaning step that fixes the
   data-quality issues flagged by validation (type coercion, canonical
   categories, duplicate removal, out-of-range values -> NaN). Because it
   learns nothing from the data it is safe to apply identically at
   training and inference time.

2. :func:`build_preprocessor` - the *stateful* scikit-learn
   ``ColumnTransformer`` (imputation + scaling + one-hot encoding). This
   object is fitted on the training split only and persisted with joblib,
   guaranteeing that inference reuses the exact same imputers, scaler and
   encoder rather than fitting new ones.

Keeping the stateful transformations inside a single ColumnTransformer is
what prevents train/inference skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    AGE_MAX,
    AGE_MIN,
    CATEGORICAL_FEATURES,
    CATEGORY_CANONICAL_MAP,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TENURE_MAX,
    TENURE_MIN,
    YES_NO_COLUMNS,
    get_logger,
)

logger = get_logger(__name__)

# Columns that arrive as text/mixed and must be coerced to numeric.
_NUMERIC_SOURCE_COLUMNS = ["age", "tenure_months", "monthly_charges", "total_charges"]


def _normalise_category(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Map trimmed/lower-cased category variants onto their canonical form.

    Missing values stay missing; values not present in the mapping keep
    their original spelling so genuinely unexpected categories remain
    visible rather than being coerced into a wrong bucket.
    """
    lowered = series.astype("string").str.strip().str.lower()
    mapped = lowered.map(mapping)
    # Fall back to the original (title-cased) value when not in the map.
    return mapped.fillna(series)


def clean_data(df: pd.DataFrame, has_target: bool = True) -> pd.DataFrame:
    """Clean a raw customer DataFrame without learning any statistics.

    Parameters
    ----------
    df:
        Raw records (post-extraction, post-validation).
    has_target:
        Whether to normalise/encode the target column. Set ``False`` for
        inference batches that carry no churn label.

    Returns
    -------
    pandas.DataFrame
        A cleaned copy. Invalid-but-recoverable values are set to ``NaN``
        so the fitted imputer can fill them consistently; nothing is
        imputed here.
    """
    df = df.copy()
    start_rows = len(df)

    # 1) Resolve duplicate customer ids (keep the first occurrence).
    if ID_COLUMN in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=ID_COLUMN, keep="first").reset_index(drop=True)
        removed = before - len(df)
        if removed:
            logger.info("Removed %d duplicate customer record(s).", removed)

    # 2) Coerce numeric-looking columns to real numbers (blanks -> NaN).
    for col in _NUMERIC_SOURCE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3) senior_citizen -> clean 0/1 integer flag.
    if "senior_citizen" in df.columns:
        df["senior_citizen"] = (
            pd.to_numeric(df["senior_citizen"], errors="coerce").fillna(0).astype(int)
        )

    # 4) Out-of-range values -> NaN (imputed later by the fitted pipeline).
    if "age" in df.columns:
        df.loc[(df["age"] < AGE_MIN) | (df["age"] > AGE_MAX), "age"] = np.nan
    if "tenure_months" in df.columns:
        df.loc[
            (df["tenure_months"] < TENURE_MIN) | (df["tenure_months"] > TENURE_MAX),
            "tenure_months",
        ] = np.nan
    for charge_col in ("monthly_charges", "total_charges"):
        if charge_col in df.columns:
            df.loc[df[charge_col] < 0, charge_col] = np.nan

    # 5) Canonicalise categorical spellings.
    if "gender" in df.columns:
        df["gender"] = _normalise_category(df["gender"], CATEGORY_CANONICAL_MAP["gender"])
    if "contract_type" in df.columns:
        df["contract_type"] = _normalise_category(
            df["contract_type"], CATEGORY_CANONICAL_MAP["contract_type"]
        )
    for col in YES_NO_COLUMNS:
        if col in df.columns:
            df[col] = _normalise_category(df[col], CATEGORY_CANONICAL_MAP["yes_no"])

    # 6) Normalise / encode the target to a clean 0/1 integer.
    if has_target and TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = _encode_target(df[TARGET_COLUMN])

    logger.info("Cleaning complete: %d -> %d rows.", start_rows, len(df))
    return df


def _encode_target(series: pd.Series) -> pd.Series:
    """Map Yes/No/1/0 (any casing) to a nullable integer 0/1 series."""
    mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    as_text = series.astype("string").str.strip().str.lower()
    encoded = as_text.map(mapping)
    # Also accept already-numeric targets.
    numeric = pd.to_numeric(series, errors="coerce")
    encoded = encoded.fillna(numeric)
    return encoded.astype("Int64")


def build_preprocessor() -> ColumnTransformer:
    """Build the (unfitted) preprocessing ColumnTransformer.

    * Numeric features: median imputation then standard scaling.
    * Categorical features: most-frequent imputation then one-hot
      encoding with ``handle_unknown='ignore'`` so unseen categories at
      inference do not break prediction.

    The returned object is fitted once on training data and saved; it is
    never refitted during inference.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor
