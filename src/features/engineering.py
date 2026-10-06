"""Feature engineering.

All engineered features are *row-local* and *stateless*: each value is
derived only from that customer's own attributes, never from the target
or from aggregate statistics of the dataset. This guarantees there is no
target leakage and that the exact same function can be applied at
inference time without fitting anything.

The function is called after cleaning and before the fitted
ColumnTransformer, in both the training and prediction paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SERVICE_COLUMNS, get_logger

logger = get_logger(__name__)

# Tenure bucket edges (months). Chosen to reflect common contract/loyalty
# milestones rather than data-driven cut points (which could leak).
_TENURE_BINS = [-0.1, 12, 24, 48, np.inf]
_TENURE_LABELS = ["0-12m", "12-24m", "24-48m", "48m+"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a cleaned DataFrame.

    New columns
    -----------
    avg_monthly_spend:
        ``total_charges / tenure`` - realised average spend, robust to the
        occasional mismatch between monthly and total charges. Falls back
        to ``monthly_charges`` when tenure is zero/unknown.
    charges_per_tenure:
        ``total_charges / (tenure + 1)`` - a leakage-free intensity
        measure that stays finite for brand-new customers.
    service_count:
        Number of add-on services the customer actively subscribes to.
    tenure_group:
        Categorical tenure bucket capturing non-linear loyalty effects.
    """
    df = df.copy()

    tenure = df["tenure_months"]
    total = df["total_charges"]
    monthly = df["monthly_charges"]

    # Average realised monthly spend (guard divide-by-zero / unknown tenure).
    safe_tenure = tenure.where(tenure.notna() & (tenure > 0))
    df["avg_monthly_spend"] = (total / safe_tenure).fillna(monthly)

    # Intensity measure that never divides by zero.
    df["charges_per_tenure"] = total / (tenure.fillna(0) + 1)

    # Count of active add-on services (treat unknowns / no-internet as 0).
    service_flags = [
        df[col].eq("Yes").astype(int) for col in SERVICE_COLUMNS if col in df.columns
    ]
    if service_flags:
        df["service_count"] = sum(service_flags)
    else:
        df["service_count"] = 0

    # Tenure bucket (stateless, fixed edges).
    df["tenure_group"] = pd.cut(
        tenure.fillna(0), bins=_TENURE_BINS, labels=_TENURE_LABELS
    ).astype("object")

    logger.info("Feature engineering added 4 columns.")
    return df
