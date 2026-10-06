"""Tests for the data validation layer."""

from __future__ import annotations

import pandas as pd

from src.config import TARGET_COLUMN
from src.data.validation import validate_customers


def test_valid_data_detects_recoverable_issues(raw_df):
    # The synthetic generator injects quality issues, so validation should
    # pass (no fatal errors) but report warnings.
    report = validate_customers(raw_df, require_target=True)
    assert report.is_valid
    assert len(report.warnings) > 0
    assert report.stats["duplicate_ids"] > 0


def test_empty_frame_is_fatal():
    report = validate_customers(pd.DataFrame(), require_target=True)
    assert not report.is_valid
    assert any("empty" in e.lower() for e in report.errors)


def test_missing_required_column_is_fatal(raw_df):
    broken = raw_df.drop(columns=["age"])
    report = validate_customers(broken, require_target=True)
    assert not report.is_valid
    assert any("age" in e for e in report.errors)


def test_missing_target_is_fatal_when_required(raw_df):
    no_target = raw_df.drop(columns=[TARGET_COLUMN])
    report = validate_customers(no_target, require_target=True)
    assert not report.is_valid


def test_missing_target_ok_for_inference(raw_df):
    no_target = raw_df.drop(columns=[TARGET_COLUMN])
    report = validate_customers(no_target, require_target=False)
    assert report.is_valid


def test_detects_out_of_range_values(raw_df):
    report = validate_customers(raw_df, require_target=True)
    assert report.stats["impossible_ages"] > 0
    assert report.stats["negative_monthly_charges"] > 0
