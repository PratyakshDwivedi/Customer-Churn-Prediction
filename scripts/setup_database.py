"""Create and seed the SQLite churn database from synthetic data.

Run this once before training::

    python scripts/setup_database.py

Steps:
    1. Generate a reproducible synthetic dataset (with deliberate
       data-quality issues).
    2. Persist it as the raw CSV (for reference only - the pipeline reads
       from SQL, not this CSV).
    3. Apply the schema and bulk-seed the records into SQLite.
    4. Emit a small 'new_customers' sample for prediction demos.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TARGET_COLUMN, config, get_logger  # noqa: E402
from src.data.generation import generate_customers  # noqa: E402
from src.database.loader import DatabaseLoader  # noqa: E402

logger = get_logger("setup_database")

SCHEMA_PATH = config.database_path.parent / "schema.sql"
N_SAMPLE_CUSTOMERS = 15


def main() -> None:
    config.ensure_directories()

    # 1) Generate the main synthetic dataset.
    df = generate_customers(n=config.n_customers, seed=config.random_seed)

    # 2) Save raw CSV for reference.
    df.to_csv(config.raw_csv_path, index=False)
    logger.info("Wrote raw CSV reference to %s", config.raw_csv_path)

    # 3) Apply schema and seed the database.
    loader = DatabaseLoader(app_config=config)
    loader.execute_script(SCHEMA_PATH)
    loader.seed_dataframe(df, if_exists="append")
    logger.info("Database now holds %d rows.", loader.row_count())

    # 4) Build a small 'unseen' batch for prediction demos (no target column).
    sample = generate_customers(n=N_SAMPLE_CUSTOMERS, seed=config.random_seed + 1)
    sample = sample.drop(columns=[TARGET_COLUMN]).head(N_SAMPLE_CUSTOMERS)
    sample.to_csv(config.sample_csv_path, index=False)
    logger.info("Wrote %d-row prediction sample to %s", len(sample), config.sample_csv_path)

    print(f"\nDatabase ready at {config.database_path} ({loader.row_count()} rows).")
    print(f"Prediction sample at {config.sample_csv_path}.")


if __name__ == "__main__":
    main()
