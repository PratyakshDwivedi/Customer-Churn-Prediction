"""Tests for the database loader and SQL extraction."""

from __future__ import annotations

import pytest

from src.data.extraction import extract_customers
from src.database.loader import DatabaseLoader


def test_connection_and_seed_roundtrip(tmp_config, raw_df):
    loader = DatabaseLoader(app_config=tmp_config)
    # Create a table directly via seeding (to_sql creates it).
    n = loader.seed_dataframe(raw_df)
    assert n == len(raw_df)
    assert loader.table_exists() is True
    assert loader.row_count() == len(raw_df)


def test_read_query_parameterised(tmp_config, raw_df):
    loader = DatabaseLoader(app_config=tmp_config)
    loader.seed_dataframe(raw_df)
    df = loader.read_query(
        f"SELECT * FROM {tmp_config.customers_table} WHERE customer_id = ?",
        params=("C000001",),
    )
    assert len(df) >= 1
    assert (df["customer_id"] == "C000001").all()


def test_extraction_returns_all_rows(tmp_config, raw_df):
    loader = DatabaseLoader(app_config=tmp_config)
    loader.seed_dataframe(raw_df)
    extracted = extract_customers(loader=loader, app_config=tmp_config)
    # Extraction reads every seeded row (duplicates included - cleaning handles them).
    assert len(extracted) == len(raw_df)
    assert "customer_id" in extracted.columns


def test_extraction_missing_table_raises(tmp_config):
    loader = DatabaseLoader(app_config=tmp_config)
    with pytest.raises(RuntimeError, match="does not exist"):
        extract_customers(loader=loader, app_config=tmp_config)


def test_row_count_rejects_unknown_table(tmp_config, raw_df):
    loader = DatabaseLoader(app_config=tmp_config)
    loader.seed_dataframe(raw_df)
    with pytest.raises(ValueError, match="Unknown table"):
        loader.row_count("customers; DROP TABLE customers")
