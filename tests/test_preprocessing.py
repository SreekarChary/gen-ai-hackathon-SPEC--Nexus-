"""Tests for Module 1 — Data Preprocessing."""
import pandas as pd
import pytest
from src.preprocessing import clean_data, engineer_features


def test_clean_data_removes_nulls():
    df = pd.DataFrame({
        "a": [1, 2, None, 4],
        "b": ["x", "y", "z", None],
    })
    result = clean_data(df)
    assert result.isnull().sum().sum() == 0


def test_clean_data_removes_duplicates():
    df = pd.DataFrame({
        "a": [1, 1, 2],
        "b": ["x", "x", "y"],
    })
    result = clean_data(df)
    assert len(result) == 2


def test_engineer_features_returns_dataframe():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    result = engineer_features(df)
    assert isinstance(result, pd.DataFrame)
