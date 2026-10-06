"""End-to-end training pipeline.

Runs the complete workflow in one command::

    python -m src.pipeline

Stages: SQL extraction -> validation -> cleaning -> feature engineering
-> stratified split -> fit preprocessor -> train models -> evaluate ->
select best -> save artifacts -> generate reports.

The run is deterministic for a fixed dataset and random seed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib

from src.config import Config, config, get_logger
from src.data.extraction import extract_customers
from src.data.preprocessing import clean_data
from src.data.validation import validate_customers
from src.features.engineering import engineer_features
from src.models.evaluate import evaluate_all, generate_all_figures
from src.models.selection import select_best_model
from src.models.train import (
    fit_preprocessor,
    split_features_target,
    train_candidate_models,
)

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Summary of a completed training run."""

    best_model_name: str
    selection_reason: str
    metrics: dict[str, dict]
    preprocessor_path: str
    model_path: str
    metrics_path: str


def run_pipeline(app_config: Config = config) -> PipelineResult:
    """Execute the full training pipeline and persist all artifacts."""
    app_config.ensure_directories()
    logger.info("=== Churn training pipeline started ===")

    # 1) Extract from SQL.
    raw = extract_customers(app_config=app_config)

    # 2) Validate (training requires a target).
    report = validate_customers(raw, require_target=True)
    report.log()
    logger.info("Validation summary: %s", report.summary())
    if not report.is_valid:
        raise RuntimeError(f"Fatal validation errors: {report.errors}")

    # 3) Clean -> 4) feature engineer.
    cleaned = clean_data(raw, has_target=True)
    engineered = engineer_features(cleaned)

    # Persist the processed dataset for transparency/reproducibility.
    processed_path = app_config.processed_data_dir / "customers_processed.csv"
    engineered.to_csv(processed_path, index=False)
    logger.info("Saved processed dataset to %s", processed_path)

    # 5) Split -> 6) fit preprocessor -> 7) train.
    data = split_features_target(engineered, app_config=app_config)
    preprocessor = fit_preprocessor(data.X_train)
    models = train_candidate_models(
        preprocessor, data.X_train, data.y_train, app_config.random_seed
    )

    # 8) Evaluate every model on the held-out test set.
    all_metrics = evaluate_all(models, preprocessor, data.X_test, data.y_test)

    # 9) Select the best model from the computed metrics.
    best_name, reason = select_best_model(all_metrics)
    best_model = models[best_name]

    # 10) Persist artifacts: preprocessor + best model.
    joblib.dump(preprocessor, app_config.preprocessor_path)
    joblib.dump(best_model, app_config.model_path)
    logger.info(
        "Saved preprocessor -> %s and best model -> %s",
        app_config.preprocessor_path,
        app_config.model_path,
    )

    # 11) Persist metrics (machine-readable) and figures.
    metrics_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": app_config.random_seed,
        "test_size": app_config.test_size,
        "n_rows_raw": int(len(raw)),
        "validation": {
            "errors": report.errors,
            "warnings": report.warnings,
            "stats": report.stats,
        },
        "models": all_metrics,
        "best_model": best_name,
        "selection_reason": reason,
    }
    app_config.metrics_path.write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )
    logger.info("Saved metrics to %s", app_config.metrics_path)

    generate_all_figures(
        models, all_metrics, preprocessor, data.X_test, data.y_test, app_config.figures_dir
    )

    logger.info("=== Pipeline complete. Best model: %s ===", best_name)
    return PipelineResult(
        best_model_name=best_name,
        selection_reason=reason,
        metrics=all_metrics,
        preprocessor_path=str(app_config.preprocessor_path),
        model_path=str(app_config.model_path),
        metrics_path=str(app_config.metrics_path),
    )


def _print_summary(result: PipelineResult) -> None:
    """Print a concise, human-readable summary of the run."""
    print("\n" + "=" * 60)
    print("CHURN PIPELINE - TRAINING SUMMARY")
    print("=" * 60)
    for name, m in result.metrics.items():
        marker = " <== selected" if name == result.best_model_name else ""
        print(
            f"{name:>20} | F1={m['f1']:.4f} | ROC-AUC={m['roc_auc']:.4f} "
            f"| Acc={m['accuracy']:.4f}{marker}"
        )
    print("-" * 60)
    print(f"Best model : {result.best_model_name}")
    print(f"Reason     : {result.selection_reason}")
    print(f"Artifacts  : {result.model_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    _print_summary(run_pipeline())
