"""End-to-end tests: train -> save artifacts -> score unseen data."""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import TARGET_COLUMN
from src.data.generation import generate_customers
from src.database.loader import DatabaseLoader
from src.models.predict import ChurnPredictor
from src.pipeline import run_pipeline


def _seed_db(tmp_config, raw_df):
    loader = DatabaseLoader(app_config=tmp_config)
    loader.seed_dataframe(raw_df)


def test_full_pipeline_creates_artifacts(tmp_config, raw_df):
    _seed_db(tmp_config, raw_df)
    result = run_pipeline(app_config=tmp_config)

    assert tmp_config.preprocessor_path.exists()
    assert tmp_config.model_path.exists()
    assert tmp_config.metrics_path.exists()
    assert result.best_model_name in {"logistic_regression", "random_forest"}
    # Figures were produced.
    assert any(tmp_config.figures_dir.glob("*.png"))


def test_prediction_on_unseen_data(tmp_config, raw_df):
    _seed_db(tmp_config, raw_df)
    run_pipeline(app_config=tmp_config)

    # Fresh, unseen batch (different seed), with no target column.
    new_batch = generate_customers(n=25, seed=999).drop(columns=[TARGET_COLUMN])

    predictor = ChurnPredictor.from_artifacts(app_config=tmp_config)
    preds = predictor.predict_dataframe(new_batch)

    assert list(preds.columns) == ["customer_id", "predicted_churn", "churn_probability"]
    assert set(preds["predicted_churn"].unique()).issubset({0, 1})
    assert preds["churn_probability"].between(0, 1).all()
    assert preds["churn_probability"].notna().all()


def test_inference_does_not_refit_preprocessor(tmp_config, raw_df):
    _seed_db(tmp_config, raw_df)
    run_pipeline(app_config=tmp_config)

    # The loaded preprocessor must already be fitted (reused, not refit).
    pre = joblib.load(tmp_config.preprocessor_path)
    # A fitted ColumnTransformer exposes fitted transformers_.
    assert hasattr(pre, "transformers_")
    new_batch = generate_customers(n=10, seed=7).drop(columns=[TARGET_COLUMN])
    predictor = ChurnPredictor.from_artifacts(app_config=tmp_config)
    # Scoring twice yields identical results (deterministic, no refit).
    first = predictor.predict_dataframe(new_batch)
    second = predictor.predict_dataframe(new_batch)
    pd.testing.assert_frame_equal(first, second)
