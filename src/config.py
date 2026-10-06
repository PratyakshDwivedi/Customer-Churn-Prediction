"""Central configuration for the churn prediction pipeline.

All tunable constants (paths, random seed, split size, model
hyper-parameters, column definitions) live here so that nothing is
hard-coded inside the processing or modelling modules. Values can be
overridden through environment variables (optionally loaded from a
local ``.env`` file) which keeps the pipeline reproducible and
configurable without editing source code.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file if present. This never raises
# when the file is missing, so it is safe in every environment.
load_dotenv()

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
# Project root = two levels up from this file (src/config.py -> project root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def _path_from_env(var: str, default: Path) -> Path:
    """Return a Path from an environment variable or fall back to a default.

    Relative values coming from the environment are resolved against the
    project root so the pipeline behaves the same regardless of the
    current working directory.
    """
    value = os.getenv(var)
    if value is None:
        return default
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


@dataclass(frozen=True)
class Config:
    """Immutable configuration object shared across the pipeline."""

    # Reproducibility -------------------------------------------------------- #
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))

    # Dataset generation ----------------------------------------------------- #
    n_customers: int = int(os.getenv("N_CUSTOMERS", "6000"))

    # Paths ------------------------------------------------------------------ #
    database_path: Path = field(
        default_factory=lambda: _path_from_env(
            "DATABASE_PATH", PROJECT_ROOT / "database" / "churn.db"
        )
    )
    raw_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed"
    )
    sample_data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "sample"
    )
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")

    # Artifact file names ---------------------------------------------------- #
    preprocessor_filename: str = "preprocessing_pipeline.joblib"
    model_filename: str = "best_model.joblib"
    metrics_filename: str = "metrics.json"

    # Database --------------------------------------------------------------- #
    customers_table: str = "customers"

    # --- Derived artifact paths -------------------------------------------- #
    @property
    def preprocessor_path(self) -> Path:
        return self.models_dir / self.preprocessor_filename

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def metrics_dir(self) -> Path:
        return self.reports_dir / "metrics"

    @property
    def metrics_path(self) -> Path:
        return self.metrics_dir / self.metrics_filename

    @property
    def raw_csv_path(self) -> Path:
        return self.raw_data_dir / "customers_raw.csv"

    @property
    def sample_csv_path(self) -> Path:
        return self.sample_data_dir / "new_customers.csv"

    def ensure_directories(self) -> None:
        """Create every output directory used by the pipeline."""
        for directory in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.sample_data_dir,
            self.models_dir,
            self.figures_dir,
            self.metrics_dir,
            self.database_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #
# Identifier and target.
ID_COLUMN: str = "customer_id"
TARGET_COLUMN: str = "churn"

# Raw columns expected straight out of the database (before feature
# engineering). Used by the validation layer to check the schema.
RAW_NUMERIC_COLUMNS: list[str] = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "senior_citizen",
]
RAW_CATEGORICAL_COLUMNS: list[str] = [
    "gender",
    "contract_type",
    "payment_method",
    "internet_service",
    "tech_support",
    "online_security",
    "partner",
    "dependents",
    "paperless_billing",
]
# Columns that must exist for the pipeline to run at all (target excluded,
# since inference data legitimately has no target).
REQUIRED_FEATURE_COLUMNS: list[str] = (
    [ID_COLUMN] + RAW_NUMERIC_COLUMNS + RAW_CATEGORICAL_COLUMNS
)

# Feature lists *after* feature engineering. These drive the
# ColumnTransformer so the preprocessing pipeline always sees the same
# columns during training and inference.
NUMERIC_FEATURES: list[str] = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "senior_citizen",
    "avg_monthly_spend",
    "charges_per_tenure",
    "service_count",
]
CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "contract_type",
    "payment_method",
    "internet_service",
    "tech_support",
    "online_security",
    "partner",
    "dependents",
    "paperless_billing",
    "tenure_group",
]
MODEL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --------------------------------------------------------------------------- #
# Domain rules used by validation and cleaning
# --------------------------------------------------------------------------- #
AGE_MIN, AGE_MAX = 18, 100
TENURE_MIN, TENURE_MAX = 0, 72

# Canonical category values and the variants that should map onto them.
# Keys are lowercase/trimmed variants; values are the canonical form.
CATEGORY_CANONICAL_MAP: dict[str, dict[str, str]] = {
    "gender": {
        "male": "Male",
        "m": "Male",
        "female": "Female",
        "f": "Female",
    },
    "contract_type": {
        "month-to-month": "Month-to-month",
        "monthly": "Month-to-month",
        "one year": "One year",
        "1 year": "One year",
        "two year": "Two year",
        "2 year": "Two year",
    },
    "yes_no": {
        "yes": "Yes",
        "y": "Yes",
        "true": "Yes",
        "1": "Yes",
        "no": "No",
        "n": "No",
        "false": "No",
        "0": "No",
        "no internet service": "No internet service",
        "no phone service": "No phone service",
    },
}
# Categorical columns that use the generic Yes/No normalisation.
YES_NO_COLUMNS: list[str] = [
    "tech_support",
    "online_security",
    "partner",
    "dependents",
    "paperless_billing",
]

# Service columns counted by the ``service_count`` engineered feature.
SERVICE_COLUMNS: list[str] = ["tech_support", "online_security"]


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a sensible default configuration.

    Configuration is applied once at the root level; repeated calls reuse
    the same handler so importing many modules does not duplicate log
    lines.
    """
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    return logger


# A single shared instance imported across the project.
config = Config()
