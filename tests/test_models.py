"""
Unit tests for turbofan_ai.models (artifact loading, inference pipelines, and boundary constraints).
"""

import numpy as np
import pytest

from turbofan_ai.models import (
    load_model_artifacts,
    predict_anomaly,
    predict_rul,
)


def test_load_model_artifacts(models_dir):
    """Verify loading real serialized artifacts from models/ directory."""
    rf, iso, feature_cols = load_model_artifacts(models_dir)
    assert hasattr(rf, "predict"), "RUL model must provide predict method"
    assert hasattr(iso, "decision_function"), "Anomaly model must provide decision_function"
    assert isinstance(feature_cols, list)
    assert len(feature_cols) == 29, f"Expected 29 trained feature columns, got {len(feature_cols)}"


def test_load_model_artifacts_missing_dir(tmp_path):
    """Verify FileNotFoundError is raised when artifacts are missing."""
    empty_dir = tmp_path / "nonexistent_models"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Required model artifact not found"):
        load_model_artifacts(empty_dir)


def test_predict_rul_non_negativity_constraint():
    """Verify physical non-negativity constraint (RUL >= 0.0) even if regressor predicts negative."""
    class MockNegativeRegressor:
        def predict(self, X):
            return np.array([50.0, 10.0, 0.0, -5.5, -120.0])

    mock_model = MockNegativeRegressor()
    X_dummy = np.zeros((5, 10))
    preds = predict_rul(mock_model, X_dummy)

    assert (preds >= 0.0).all(), "Predicted RUL values must never be negative"
    assert preds[3] == 0.0
    assert preds[4] == 0.0
    assert preds[0] == 50.0


def test_predict_rul_real_model(models_dir, sample_dataframe):
    """Verify RUL inference against real model artifact on sample dataset."""
    rf, _, feature_cols = load_model_artifacts(models_dir)
    X = sample_dataframe[feature_cols]
    preds = predict_rul(rf, X)

    assert len(preds) == len(sample_dataframe)
    assert (preds >= 0.0).all()
    assert np.isfinite(preds).all()


def test_predict_anomaly_thresholding(models_dir, sample_dataframe):
    """Verify anomaly score evaluation and dynamic sensitivity thresholding."""
    _, iso, feature_cols = load_model_artifacts(models_dir)
    X = sample_dataframe[feature_cols]

    scores_0, flags_0 = predict_anomaly(iso, X, threshold=0.0)
    assert len(scores_0) == len(sample_dataframe)
    assert set(np.unique(flags_0)).issubset({-1, 1})

    # Verify flags match threshold comparison exactly
    expected_flags_0 = np.where(scores_0 < 0.0, -1, 1)
    np.testing.assert_array_equal(flags_0, expected_flags_0)

    # Tuning threshold downward (-0.1) should strictly reduce or maintain anomaly count
    _, flags_tuned = predict_anomaly(iso, X, threshold=-0.1)
    anom_count_0 = (flags_0 == -1).sum()
    anom_count_tuned = (flags_tuned == -1).sum()
    assert anom_count_tuned <= anom_count_0, "Lowering sensitivity threshold must not increase anomaly detections"
