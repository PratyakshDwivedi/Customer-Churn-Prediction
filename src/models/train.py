"""Model training.

Defines the candidate models, the stratified train/test split, and the
fitting routine. A single preprocessing ColumnTransformer is fitted once
on the training split and shared by every candidate model, which keeps
the comparison fair and lets the fitted preprocessor be persisted on its
own for reuse at inference time.

Class imbalance (~27% churn) is handled with ``class_weight='balanced'``
- a lightweight, well-understood technique appropriate for this moderate
skew, avoiding the extra complexity and resampling risks of SMOTE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import MODEL_FEATURES, TARGET_COLUMN, Config, config, get_logger
from src.data.preprocessing import build_preprocessor

logger = get_logger(__name__)


def get_model_definitions(seed: int) -> dict[str, ClassifierMixin]:
    """Return the candidate classifiers, keyed by display name.

    Both use ``class_weight='balanced'`` and a fixed ``random_state`` for
    reproducibility.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }


@dataclass
class TrainingData:
    """Container for the split feature/target arrays."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_features_target(
    df: pd.DataFrame, app_config: Config = config
) -> TrainingData:
    """Split an engineered DataFrame into stratified train/test sets.

    Rows with a missing target cannot be used for supervised training and
    are dropped here (and only here), after being surfaced by validation.
    """
    usable = df.dropna(subset=[TARGET_COLUMN]).copy()
    dropped = len(df) - len(usable)
    if dropped:
        logger.warning("Dropped %d row(s) with missing target before split.", dropped)

    X = usable[MODEL_FEATURES]
    y = usable[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=app_config.test_size,
        random_state=app_config.random_seed,
        stratify=y,
    )
    logger.info(
        "Split data: %d train / %d test rows (churn rate train=%.1f%%, test=%.1f%%)",
        len(X_train),
        len(X_test),
        100 * y_train.mean(),
        100 * y_test.mean(),
    )
    return TrainingData(X_train, X_test, y_train, y_test)


def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit the preprocessing ColumnTransformer on the training split only."""
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    n_features = preprocessor.transform(X_train.head(1)).shape[1]
    logger.info("Fitted preprocessor -> %d engineered model features.", n_features)
    return preprocessor


def train_candidate_models(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> dict[str, ClassifierMixin]:
    """Fit every candidate model on the preprocessed training data.

    The preprocessor is applied (not refitted) to produce the feature
    matrix every model is trained on.
    """
    X_train_t = preprocessor.transform(X_train)
    fitted: dict[str, ClassifierMixin] = {}
    for name, estimator in get_model_definitions(seed).items():
        model = clone(estimator)
        model.fit(X_train_t, np.asarray(y_train))
        fitted[name] = model
        logger.info("Trained model '%s'.", name)
    return fitted
