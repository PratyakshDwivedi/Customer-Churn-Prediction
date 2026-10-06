"""Model evaluation and visualisation.

Computes the full metric set (F1 and ROC-AUC as the primary metrics, plus
accuracy, precision, recall and the confusion matrix) from *actual*
model predictions on the held-out test set. Nothing is hard-coded; every
number comes from scikit-learn. Figures (confusion matrices, ROC curves,
model comparison) are written to ``reports/figures``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend - safe on a laptop/CI with no display.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.base import ClassifierMixin  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import get_logger  # noqa: E402

logger = get_logger(__name__)


def evaluate_model(
    model: ClassifierMixin,
    preprocessor: ColumnTransformer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Compute metrics for one fitted model on the test set.

    Returns a JSON-serialisable dict containing F1, ROC-AUC, accuracy,
    precision, recall and the confusion matrix.
    """
    y_true = np.asarray(y_test).astype(int)
    X_test_t = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_t).astype(int)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(y_true)),
    }
    logger.info(
        "Evaluated model: F1=%.4f ROC-AUC=%.4f Acc=%.4f",
        metrics["f1"],
        metrics["roc_auc"],
        metrics["accuracy"],
    )
    return metrics


def evaluate_all(
    models: dict[str, ClassifierMixin],
    preprocessor: ColumnTransformer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict]:
    """Evaluate every fitted model and return a {name: metrics} dict."""
    return {
        name: evaluate_model(model, preprocessor, X_test, y_test)
        for name, model in models.items()
    }


# --------------------------------------------------------------------------- #
# Visualisations
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    metrics: dict, name: str, out_dir: Path
) -> Path:
    """Save a labelled confusion-matrix heatmap for one model."""
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    path = out_dir / f"confusion_matrix_{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc_curves(
    models: dict[str, ClassifierMixin],
    preprocessor: ColumnTransformer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path,
) -> Path:
    """Overlay ROC curves for all models on a single axis."""
    y_true = np.asarray(y_test).astype(int)
    X_test_t = preprocessor.transform(X_test)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / "roc_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_model_comparison(all_metrics: dict[str, dict], out_dir: Path) -> Path:
    """Save a grouped bar chart comparing key metrics across models."""
    metric_keys = ["f1", "roc_auc", "accuracy", "precision", "recall"]
    names = list(all_metrics.keys())
    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(names):
        values = [all_metrics[name][k] for k in metric_keys]
        ax.bar(x + i * width, values, width, label=name)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([k.upper() for k in metric_keys])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "model_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_all_figures(
    models: dict[str, ClassifierMixin],
    all_metrics: dict[str, dict],
    preprocessor: ColumnTransformer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path,
) -> list[Path]:
    """Generate every evaluation figure and return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [plot_confusion_matrix(m, name, out_dir) for name, m in all_metrics.items()]
    paths.append(plot_roc_curves(models, preprocessor, X_test, y_test, out_dir))
    paths.append(plot_model_comparison(all_metrics, out_dir))
    logger.info("Wrote %d evaluation figure(s) to %s", len(paths), out_dir)
    return paths
