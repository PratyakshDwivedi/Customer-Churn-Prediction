# Customer Churn Prediction Pipeline

An end-to-end, production-style machine learning pipeline that predicts
customer churn. Raw records live in a **SQL database**; the pipeline
extracts them, validates and cleans the data, engineers features, trains
and compares two classifiers, automatically selects the best one, and
persists reusable artifacts so new customer batches can be scored
**without retraining**.

> All metrics in this README are produced by actually running the
> pipeline (`reports/metrics/metrics.json`). None are hand-written.

## Overview

The goal is a reproducible churn classifier that mirrors how such a
system is built in practice: SQL as the source of truth, a strict
train/inference contract (the same cleaning, feature engineering and
*fitted* preprocessing are reused at prediction time), automated model
selection from real metrics, and a full test suite.

## Architecture

```
          ┌─────────────────┐
          │   SQL Database  │   (SQLite)
          └────────┬────────┘
                   ↓
            Data Extraction      src/data/extraction.py
                   ↓
            Data Validation      src/data/validation.py   (fatal vs recoverable)
                   ↓
             Data Cleaning       src/data/preprocessing.py:clean_data
                   ↓
          Feature Engineering    src/features/engineering.py
                   ↓
   ┌──────────────────────────┐
   │     ColumnTransformer     │  src/data/preprocessing.py:build_preprocessor
   │  + scikit-learn Pipeline  │  (impute + scale + one-hot)  ← fitted once, saved
   └───────────┬──────────────┘
               ↓
     ┌─────────┴─────────┐
     ↓                   ↓
Logistic Regression   Random Forest     src/models/train.py
     ↓                   ↓
     └─────────┬─────────┘
               ↓
       Model Evaluation           src/models/evaluate.py   (F1 + ROC-AUC + more)
               ↓
       Model Selection            src/models/selection.py  (automated, documented rule)
               ↓
   Saved Best Model + Saved Preprocessor   (models/*.joblib)
               ↓
       New Customer Data
               ↓
         Prediction               src/models/predict.py
```

The **training** and **inference** paths deliberately share the exact
same cleaning and feature-engineering functions, and inference loads the
*already-fitted* preprocessor rather than fitting a new one:

```
TRAINING:   Raw SQL → clean → engineer → FIT preprocessor → train models
INFERENCE:  New raw → clean → engineer → APPLY saved preprocessor → saved model → predict
```

## Features

- SQL-backed ingestion with a reusable, injection-safe database loader.
- Validation layer that separates **fatal errors** (stop the run) from
  **recoverable data-quality issues** (reported, then cleaned).
- Stateless cleaning + leakage-free feature engineering, reused verbatim
  at inference.
- scikit-learn `ColumnTransformer` / `Pipeline` for imputation, scaling
  and one-hot encoding, persisted with joblib.
- Two models trained and compared; the winner chosen automatically from
  computed metrics.
- Machine-readable metrics (`JSON`) plus confusion-matrix, ROC and
  comparison figures.
- Batch prediction CLI that scores unseen customers without retraining.
- Full `pytest` suite and environment-variable-based configuration.

## Tech Stack

Python · SQL (SQLite) · pandas · NumPy · scikit-learn · joblib ·
Matplotlib · seaborn · pytest · python-dotenv

## Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/                 # CSV reference copy of the generated data
│   ├── processed/           # processed dataset + prediction outputs
│   └── sample/              # new_customers.csv for prediction demos
├── database/
│   ├── schema.sql           # customers table definition
│   ├── seed.sql             # example INSERTs (full seed is programmatic)
│   └── churn.db             # SQLite database (generated)
├── src/
│   ├── config.py            # central configuration (env-overridable)
│   ├── database/loader.py   # reusable SQLite access layer
│   ├── data/
│   │   ├── generation.py    # synthetic dataset generator
│   │   ├── extraction.py    # SQL → DataFrame
│   │   ├── validation.py    # validation report (fatal vs recoverable)
│   │   └── preprocessing.py # clean_data + build_preprocessor
│   ├── features/engineering.py
│   ├── models/
│   │   ├── train.py         # split, fit preprocessor, train models
│   │   ├── evaluate.py      # metrics + figures
│   │   ├── selection.py     # automated model selection
│   │   └── predict.py       # load artifacts & score new data
│   └── pipeline.py          # end-to-end orchestration
├── models/                  # saved preprocessor + best model (generated)
├── reports/
│   ├── figures/             # confusion matrices, ROC, comparison (generated)
│   └── metrics/metrics.json # machine-readable metrics (generated)
├── scripts/
│   ├── setup_database.py    # generate data + seed the database
│   ├── train.py             # run the training pipeline
│   └── predict.py           # score a new batch
├── tests/                   # pytest suite
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Dataset

The dataset is **synthetic and reproducible** (no proprietary data is
used). `src/data/generation.py` builds ~6,000 customers from a fixed
seed, with:

- a genuine but **noisy** logistic relationship between features and
  churn (so performance is realistic, not artificially perfect);
- deliberate **data-quality issues** the pipeline must handle: missing
  values, duplicate customer IDs, impossible ages, negative charges,
  `total_charges` stored as text with blanks, and inconsistent category
  spellings.

Attributes include demographics (`age`, `gender`, `senior_citizen`,
`partner`, `dependents`), account details (`tenure_months`,
`contract_type`, `payment_method`, `internet_service`, `tech_support`,
`online_security`, `paperless_billing`), charges (`monthly_charges`,
`total_charges`) and the target `churn`.

## Database

Records are stored in a SQLite table `customers` (see
`database/schema.sql`). `total_charges` is typed as text and
`customer_id` is intentionally **not** unique, mirroring a messy
production export so that data quality is enforced in code, not hidden by
the database. `src/database/loader.py` provides a reusable loader with a
context-managed connection, parameterized reads, and a table-name
allow-list to prevent SQL injection. The pipeline reads from SQL — never
directly from the CSV.

## ML Pipeline

**Cleaning** (stateless): resolves duplicate IDs, coerces numeric columns
(blanks → `NaN`), canonicalises category spellings, sets out-of-range
values (impossible ages, negative charges) to `NaN`, and encodes the
target to 0/1. It learns no statistics, so it is safe to reuse at
inference.

**Feature engineering** (leakage-free, row-local): `avg_monthly_spend`,
`charges_per_tenure`, `service_count`, and a categorical `tenure_group`.
No feature uses the target or dataset-wide aggregates.

**Preprocessing** (`ColumnTransformer`): numeric features → median
imputation + standard scaling; categorical features → most-frequent
imputation + one-hot encoding (`handle_unknown='ignore'`). It is fitted
**once** on the training split and saved; inference reuses it unchanged.

## Models

- **Logistic Regression** — linear baseline, `class_weight='balanced'`.
- **Random Forest** — non-linear ensemble (300 trees, depth-limited),
  `class_weight='balanced_subsample'`.

Class imbalance (~32% churn) is handled with class weighting plus a
stratified split — a deliberately simple, well-understood approach rather
than SMOTE, which is not warranted at this moderate skew.

## Evaluation

Primary metrics are **F1-score** and **ROC-AUC**; accuracy, precision,
recall and the confusion matrix are also computed on a held-out test set
(1,200 rows). Results from the latest run (seed 42):

| Model               |     F1 | ROC-AUC | Accuracy | Precision | Recall |
|---------------------|-------:|--------:|---------:|----------:|-------:|
| Logistic Regression | 0.5751 |  0.7470 |   0.6700 |    0.4917 | 0.6925 |
| Random Forest       | 0.5711 |  0.7446 |   0.6908 |    0.5167 | 0.6382 |

Figures are written to `reports/figures/` (per-model confusion matrices,
overlaid ROC curves, and a model-comparison bar chart).

### Model selection

Selection is automated and based solely on the computed metrics (see
`src/models/selection.py`):

1. **Primary metric: F1.** The best F1 defines the bar.
2. **Tie-break: ROC-AUC.** Models whose F1 is within `0.01` of the best
   are treated as tied; among them the highest ROC-AUC wins.

On the latest run both models fell within the F1 tolerance, so the
tie-break applied and **Logistic Regression** was selected on its higher
ROC-AUC (0.7470 vs 0.7446).

## Installation

```bash
# From the project root
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.10+ is required (built and verified on 3.11).

## Usage

```bash
# 1. Create and seed the SQLite database (also writes a prediction sample)
python scripts/setup_database.py

# 2. Train, evaluate, select, and save artifacts
python -m src.pipeline
#   (equivalent: python scripts/train.py)

# 3. Score a new batch of customers (no retraining)
python scripts/predict.py --input data/sample/new_customers.csv

# 4. Run the test suite
pytest
```

## Example Prediction

Actual output from `python scripts/predict.py --input
data/sample/new_customers.csv` (first rows of
`data/processed/predictions.csv`):

```
customer_id,predicted_churn,churn_probability
C000001,0,0.4851
C000002,1,0.5408
C000003,1,0.5362
C000004,1,0.6872
C000005,1,0.7906
C000006,0,0.1858
```

## Reproducibility

- A fixed `RANDOM_SEED` (default 42) controls data generation, the
  train/test split, and the models.
- Configuration (seed, split size, dataset size, paths) is centralised in
  `src/config.py` and overridable via environment variables / `.env`.
- To rerun on new data, replace the records in the `customers` table (or
  point `DATABASE_PATH` at another SQLite file with the same schema) and
  rerun `python -m src.pipeline`. Scoring a different batch only requires
  a new CSV passed to `scripts/predict.py` — no code changes.

## Future Improvements

These are **not yet implemented**; they are realistic next steps:

- Hyperparameter tuning (e.g. grid/Bayesian search).
- Cross-validation for more robust metric estimates.
- Additional models (gradient boosting) in the comparison.
- Experiment tracking (e.g. MLflow) and model/data versioning.
- Data-drift detection and model monitoring in production.
- A REST API / batch service for deployment.
