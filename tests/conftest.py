"""Shared pytest fixtures.

Fixtures build small, fast, fully in-memory/temp-dir versions of the
pipeline inputs so tests never touch the real project database or
artifacts.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from src.config import Config, config
from src.data.generation import generate_customers

SEED = 42
N_ROWS = 800


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """A small synthetic raw dataset with deliberate quality issues."""
    return generate_customers(n=N_ROWS, seed=SEED)


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    """A Config pointing every output path into a temp directory."""
    cfg = dataclasses.replace(
        config,
        database_path=tmp_path / "db" / "churn.db",
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        sample_data_dir=tmp_path / "data" / "sample",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        n_customers=N_ROWS,
    )
    cfg.ensure_directories()
    return cfg
