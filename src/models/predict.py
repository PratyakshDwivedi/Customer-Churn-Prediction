"""Batch prediction on new, unseen customer data.

Loads the persisted preprocessor and best model and scores a fresh batch
*without any retraining or refitting*. The batch travels through the
identical cleaning and feature-engineering functions used in training,
then through the saved (already-fitted) ColumnTransformer, guaranteeing
no train/inference skew.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer

from src.config import (
    ID_COLUMN,
    MODEL_FEATURES,
    Config,
    config,
    get_logger,
)
from src.data.preprocessing import clean_data
from src.data.validation import validate_customers
from src.features.engineering import engineer_features

logger = get_logger(__name__)


class ChurnPredictor:
    """Load persisted artifacts and score new customer batches."""

    def __init__(
        self,
        preprocessor: ColumnTransformer,
        model: ClassifierMixin,
    ):
        self.preprocessor = preprocessor
        self.model = model

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_artifacts(cls, app_config: Config = config) -> "ChurnPredictor":
        """Build a predictor from the saved joblib artifacts."""
        if not app_config.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {app_config.preprocessor_path}. "
                "Train the pipeline first."
            )
        if not app_config.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {app_config.model_path}. Train the pipeline first."
            )
        preprocessor = joblib.load(app_config.preprocessor_path)
        model = joblib.load(app_config.model_path)
        logger.info("Loaded preprocessor and model artifacts.")
        return cls(preprocessor=preprocessor, model=model)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Score a raw customer DataFrame.

        Returns a DataFrame with columns ``customer_id``,
        ``predicted_churn`` (0/1) and ``churn_probability`` (float).
        """
        # Validate (target not required for inference); fatal errors stop us.
        report = validate_customers(raw_df, require_target=False)
        report.log()
        if not report.is_valid:
            raise ValueError(
                f"New-data validation failed: {report.errors}"
            )

        cleaned = clean_data(raw_df, has_target=False)
        engineered = engineer_features(cleaned)

        X = engineered[MODEL_FEATURES]
        X_t = self.preprocessor.transform(X)

        predictions = self.model.predict(X_t).astype(int)
        probabilities = self.model.predict_proba(X_t)[:, 1]

        result = pd.DataFrame(
            {
                ID_COLUMN: engineered[ID_COLUMN].to_numpy(),
                "predicted_churn": predictions,
                "churn_probability": probabilities.round(4),
            }
        )
        logger.info(
            "Scored %d customer(s); %d predicted to churn.",
            len(result),
            int(predictions.sum()),
        )
        return result

    def predict_csv(self, input_path: Path | str, output_path: Path | str) -> pd.DataFrame:
        """Score a CSV file and write predictions to ``output_path``."""
        input_path = Path(input_path)
        raw_df = pd.read_csv(input_path)
        result = self.predict_dataframe(raw_df)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        logger.info("Wrote predictions to %s", output_path)
        return result
