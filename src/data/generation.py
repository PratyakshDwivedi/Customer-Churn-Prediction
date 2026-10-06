"""Synthetic customer churn dataset generation.

The project ships no proprietary data, so a *reproducible synthetic*
dataset is generated instead. The generator deliberately:

* builds a realistic (but noisy) relationship between features and churn
  so models learn a genuine, non-trivial signal rather than a perfect
  rule;
* injects the kind of data-quality problems the validation and cleaning
  layers exist to handle (missing values, duplicate ids, impossible
  ages, negative charges, inconsistent category spellings, numeric
  columns stored as text).

Everything is driven by a fixed seed so the dataset is identical on every
run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN, get_logger

logger = get_logger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_customers(n: int, seed: int) -> pd.DataFrame:
    """Generate a synthetic customer dataset with a realistic churn signal.

    Parameters
    ----------
    n:
        Number of *clean* customers to generate before quality issues and
        duplicates are injected (the returned frame is slightly larger
        because duplicates are appended).
    seed:
        Random seed controlling every stochastic choice.

    Returns
    -------
    pandas.DataFrame
        Raw-looking customer records including deliberate data-quality
        issues. ``total_charges`` is returned as an object/text column to
        mimic the well-known Telco dataset.
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic customers (seed=%d)", n, seed)

    # --- Base demographic / account attributes ---------------------------- #
    age = rng.integers(18, 85, size=n)
    gender = rng.choice(["Male", "Female"], size=n)
    senior_citizen = (age >= 65).astype(int)
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    tenure_months = rng.integers(0, 73, size=n)

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.55, 0.25, 0.20],
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n,
        p=[0.35, 0.22, 0.22, 0.21],
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.45, 0.20]
    )

    # Support/security availability depends on having internet.
    has_internet = internet_service != "No"
    tech_support = np.where(
        has_internet,
        rng.choice(["Yes", "No"], size=n, p=[0.40, 0.60]),
        "No internet service",
    )
    online_security = np.where(
        has_internet,
        rng.choice(["Yes", "No"], size=n, p=[0.37, 0.63]),
        "No internet service",
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])

    # --- Charges ----------------------------------------------------------- #
    # Monthly charge driven by internet tier plus noise.
    base_charge = np.select(
        [internet_service == "Fiber optic", internet_service == "DSL"],
        [80.0, 55.0],
        default=20.0,
    )
    monthly_charges = np.round(
        base_charge + rng.normal(0, 8, size=n) + (tech_support == "Yes") * 6, 2
    )
    monthly_charges = np.clip(monthly_charges, 15.0, 130.0)
    # Total charges roughly tenure * monthly with some drift.
    total_charges = np.round(
        monthly_charges * np.maximum(tenure_months, 1)
        + rng.normal(0, 25, size=n),
        2,
    )
    total_charges = np.clip(total_charges, 0.0, None)

    # --- Churn signal (noisy logistic model, no leakage) ------------------- #
    logit = (
        -1.15
        + 0.95 * (contract_type == "Month-to-month")
        - 0.35 * (contract_type == "Two year")
        - 0.045 * tenure_months
        + 0.015 * monthly_charges
        + 0.55 * (internet_service == "Fiber optic")
        - 0.45 * (tech_support == "Yes")
        - 0.35 * (online_security == "Yes")
        + 0.45 * (payment_method == "Electronic check")
        + 0.30 * senior_citizen
        - 0.20 * (partner == "Yes")
        - 0.15 * (dependents == "Yes")
        + 0.30 * (paperless_billing == "Yes")
    )
    # Irreducible noise keeps model performance realistic (AUC ~0.83).
    logit = logit + rng.normal(0, 1.1, size=n)
    churn_prob = _sigmoid(logit)
    churn = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{idx:06d}" for idx in range(1, n + 1)],
            "age": age,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "online_security": online_security,
            "paperless_billing": paperless_billing,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            TARGET_COLUMN: np.where(churn == 1, "Yes", "No"),
        }
    )

    df = _inject_quality_issues(df, rng)
    logger.info(
        "Generated %d rows; churn rate (clean signal) = %.1f%%",
        len(df),
        100 * churn.mean(),
    )
    return df


def _inject_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Introduce controlled, realistic data-quality problems.

    The issues mirror what a validation/cleaning layer must cope with in
    production feeds and are what the tests assert get handled.
    """
    df = df.copy()
    n = len(df)

    # 1) total_charges as text with some blank strings (classic Telco quirk).
    total_as_text = df["total_charges"].astype(str)
    blank_idx = rng.choice(n, size=max(1, int(0.01 * n)), replace=False)
    total_as_text.iloc[blank_idx] = " "  # whitespace, not empty
    df["total_charges"] = total_as_text

    # 2) Missing monthly_charges.
    miss_monthly = rng.choice(n, size=int(0.02 * n), replace=False)
    df.loc[df.index[miss_monthly], "monthly_charges"] = np.nan

    # 3) Missing categorical values (tech_support).
    miss_tech = rng.choice(n, size=int(0.015 * n), replace=False)
    df.loc[df.index[miss_tech], "tech_support"] = np.nan

    # 4) Inconsistent category spelling / casing.
    noisy_gender = rng.choice(n, size=int(0.03 * n), replace=False)
    df.loc[df.index[noisy_gender], "gender"] = rng.choice(
        ["male", "FEMALE", "m", "f"], size=len(noisy_gender)
    )
    noisy_contract = rng.choice(n, size=int(0.02 * n), replace=False)
    df.loc[df.index[noisy_contract], "contract_type"] = rng.choice(
        ["month-to-month", "two year", "1 year"], size=len(noisy_contract)
    )

    # 5) Impossible ages.
    bad_age = rng.choice(n, size=int(0.005 * n), replace=False)
    df.loc[df.index[bad_age], "age"] = rng.choice([-5, 0, 150, 200], size=len(bad_age))

    # 6) Negative charges.
    neg_charge = rng.choice(n, size=int(0.005 * n), replace=False)
    df.loc[df.index[neg_charge], "monthly_charges"] = -df.loc[
        df.index[neg_charge], "monthly_charges"
    ].abs()

    # 7) Duplicate customer ids (append a handful of exact-id duplicates).
    dup_rows = df.sample(n=max(2, int(0.004 * n)), random_state=0).copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df
