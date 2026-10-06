"""Automated model selection.

The winner is chosen purely from the metrics computed in
:mod:`src.models.evaluate` - never hard-coded. The rule is explicit:

1. **Primary metric: F1-score.** The model with the highest test F1 wins.
2. **Tie-breaker: ROC-AUC.** If two models' F1 scores are within
   ``F1_TOLERANCE`` of each other they are treated as tied, and the one
   with the higher ROC-AUC is selected.

This favours a model that balances precision and recall on the churn
class, while using the ranking-quality metric to break near-ties.
"""

from __future__ import annotations

from src.config import get_logger

logger = get_logger(__name__)

# Two models whose F1 differs by less than this are considered tied.
F1_TOLERANCE: float = 0.01


def select_best_model(all_metrics: dict[str, dict]) -> tuple[str, str]:
    """Return ``(best_model_name, human_readable_reason)``.

    Parameters
    ----------
    all_metrics:
        Mapping of model name -> metrics dict (must contain ``f1`` and
        ``roc_auc``).
    """
    if not all_metrics:
        raise ValueError("No model metrics provided to selection.")

    if len(all_metrics) == 1:
        only_name, only_metrics = next(iter(all_metrics.items()))
        reason = f"Only candidate; F1={only_metrics['f1']:.4f}."
        logger.info("Selected best model '%s': %s", only_name, reason)
        return only_name, reason

    # Step 1: the best achievable F1.
    best_f1 = max(m["f1"] for m in all_metrics.values())

    # Step 2: every model whose F1 is within tolerance of the best is a
    # "contender". A single clear winner leaves one contender; near-ties
    # leave several.
    contenders = {
        name: m
        for name, m in all_metrics.items()
        if best_f1 - m["f1"] < F1_TOLERANCE
    }

    # Step 3: among contenders, the highest ROC-AUC wins (this is the
    # tie-break, and is a no-op when there is a single clear F1 leader).
    best_name = max(contenders, key=lambda n: contenders[n]["roc_auc"])
    best_metrics = all_metrics[best_name]

    if len(contenders) > 1:
        reason = (
            f"{len(contenders)} model(s) within F1 tolerance ({F1_TOLERANCE}) "
            f"of the best F1 ({best_f1:.4f}); selected on highest ROC-AUC "
            f"({best_metrics['roc_auc']:.4f})."
        )
    else:
        runner_f1 = max(
            m["f1"] for n, m in all_metrics.items() if n != best_name
        )
        reason = (
            f"Highest F1-score ({best_metrics['f1']:.4f}, next best "
            f"{runner_f1:.4f}, gap {best_metrics['f1'] - runner_f1:.4f})."
        )

    logger.info("Selected best model '%s': %s", best_name, reason)
    return best_name, reason
