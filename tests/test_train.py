"""Tests for Module 2 — Model Training."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from src.train import train_random_forest, train_xgboost, evaluate


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=200, n_features=10,
        n_informative=5, random_state=42,
    )
    return X, y


def test_random_forest_trains(sample_data):
    X, y = sample_data
    model = train_random_forest(X, y)
    assert hasattr(model, "predict")


def test_xgboost_trains(sample_data):
    X, y = sample_data
    model = train_xgboost(X, y)
    assert hasattr(model, "predict")


def test_evaluate_returns_metrics(sample_data):
    X, y = sample_data
    model = train_random_forest(X, y)
    metrics = evaluate(model, X, y, "test-rf")
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
