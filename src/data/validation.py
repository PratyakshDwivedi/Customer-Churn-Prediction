"""Data validation layer.

Validation runs *before* cleaning and classifies problems into two
buckets:

* **Fatal errors** - the data cannot be used at all (missing required
  columns, empty frame, a completely unusable target). These stop the
  pipeline.
* **Recoverable issues** - real data-quality problems (missing values,
  duplicate ids, out-of-range numbers, unknown categories) that the
  cleaning layer is designed to fix. These are reported as warnings, not
  silently ignored.

The layer never drops or mutates data itself; it only inspects and
reports, so the same report can be logged, asserted on in tests, or
surfaced to a user.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import (
    AGE_MAX,
    AGE_MIN,
    ID_COLUMN,
    REQUIRED_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TENURE_MAX,
    TENURE_MIN,
    get_logger,
)

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Structured result of validating a raw customer DataFrame."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """``True`` when there are no fatal errors (warnings are allowed)."""
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def log(self) -> None:
        """Emit the report through the logging system."""
        for message in self.errors:
            logger.error("VALIDATION ERROR: %s", message)
        for message in self.warnings:
            logger.warning("DATA QUALITY: %s", message)
        if self.is_valid and not self.warnings:
            logger.info("Validation passed with no issues.")

    def summary(self) -> str:
        return (
            f"{len(self.errors)} fatal error(s), {len(self.warnings)} "
            f"recoverable issue(s)."
        )


def _to_numeric(series: pd.Series) -> pd.Series:
    """Best-effort numeric coercion used for inspection only."""
    return pd.to_numeric(series, errors="coerce")


def validate_customers(df: pd.DataFrame, require_target: bool = True) -> ValidationReport:
    """Validate a raw customer DataFrame and return a report.

    Parameters
    ----------
    df:
        Raw records straight from extraction.
    require_target:
        When ``True`` (training) the target column must be present and
        usable. For inference batches it is optional.
    """
    report = ValidationReport()
    report.stats["n_rows"] = len(df)

    # --- Fatal: structural checks ----------------------------------------- #
    if df.empty:
        report.add_error("DataFrame is empty - no records to process.")
        return report

    missing_cols = [c for c in REQUIRED_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        report.add_error(f"Missing required columns: {missing_cols}")
        # Without the required columns, further checks are unreliable.
        return report

    if require_target:
        if TARGET_COLUMN not in df.columns:
            report.add_error(f"Required target column '{TARGET_COLUMN}' is missing.")
            return report
        target_valid = df[TARGET_COLUMN].isin(["Yes", "No", 0, 1, "0", "1"])
        if target_valid.sum() == 0:
            report.add_error(
                f"Target column '{TARGET_COLUMN}' has no recognisable values."
            )
            return report

    # --- Recoverable: duplicate ids --------------------------------------- #
    dup_ids = int(df[ID_COLUMN].duplicated().sum())
    report.stats["duplicate_ids"] = dup_ids
    if dup_ids:
        report.add_warning(f"{dup_ids} duplicate '{ID_COLUMN}' value(s) found.")

    # --- Recoverable: missing values -------------------------------------- #
    missing_counts = df.isna().sum()
    total_missing = int(missing_counts.sum())
    # total_charges blanks are whitespace strings, not NaN - count them too.
    if "total_charges" in df.columns:
        blank_total = int(
            (_to_numeric(df["total_charges"]).isna() & df["total_charges"].notna()).sum()
        )
        report.stats["unparseable_total_charges"] = blank_total
        if blank_total:
            report.add_warning(
                f"{blank_total} 'total_charges' value(s) are non-numeric/blank."
            )
    report.stats["missing_values"] = total_missing
    if total_missing:
        cols_with_missing = missing_counts[missing_counts > 0].to_dict()
        report.add_warning(f"Missing values present: {cols_with_missing}")

    # --- Recoverable: out-of-range numerics ------------------------------- #
    age_num = _to_numeric(df["age"])
    bad_age = int(((age_num < AGE_MIN) | (age_num > AGE_MAX)).sum())
    report.stats["impossible_ages"] = bad_age
    if bad_age:
        report.add_warning(
            f"{bad_age} age value(s) outside [{AGE_MIN}, {AGE_MAX}]."
        )

    tenure_num = _to_numeric(df["tenure_months"])
    bad_tenure = int(((tenure_num < TENURE_MIN) | (tenure_num > TENURE_MAX)).sum())
    report.stats["invalid_tenure"] = bad_tenure
    if bad_tenure:
        report.add_warning(
            f"{bad_tenure} tenure value(s) outside [{TENURE_MIN}, {TENURE_MAX}]."
        )

    for charge_col in ("monthly_charges", "total_charges"):
        if charge_col in df.columns:
            charge_num = _to_numeric(df[charge_col])
            negatives = int((charge_num < 0).sum())
            report.stats[f"negative_{charge_col}"] = negatives
            if negatives:
                report.add_warning(f"{negatives} negative value(s) in '{charge_col}'.")

    return report
