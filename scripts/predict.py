"""Score a batch of new customers with the trained model.

Usage::

    python scripts/predict.py --input data/sample/new_customers.csv
    python scripts/predict.py --input my_batch.csv --output preds.csv

Loads the saved preprocessor and best model (no retraining), applies the
same cleaning/feature-engineering/preprocessing, and writes a CSV of
``customer_id,predicted_churn,churn_probability``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import config, get_logger  # noqa: E402
from src.models.predict import ChurnPredictor  # noqa: E402

logger = get_logger("predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict customer churn on a new batch.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a CSV of new customer records.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write predictions (default: data/processed/predictions.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else config.processed_data_dir / "predictions.csv"
    )

    predictor = ChurnPredictor.from_artifacts(app_config=config)
    result = predictor.predict_csv(input_path, output_path)

    print(f"\nPredictions written to {output_path}\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
