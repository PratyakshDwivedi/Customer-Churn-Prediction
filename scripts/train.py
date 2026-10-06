"""Train the churn pipeline and persist artifacts.

Thin wrapper around :func:`src.pipeline.run_pipeline` so training can be
launched either as ``python scripts/train.py`` or ``python -m
src.pipeline``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import _print_summary, run_pipeline  # noqa: E402


def main() -> None:
    _print_summary(run_pipeline())


if __name__ == "__main__":
    main()
