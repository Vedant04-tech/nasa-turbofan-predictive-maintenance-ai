"""
Unit tests for turbofan_ai.evaluation (regression metrics, prognostic lead times, and health summaries).
"""

import numpy as np
import pandas as pd
import pytest

from turbofan_ai.evaluation import (
    calculate_early_warning_lead_time,
    compute_regression_metrics,
    get_anomaly_summary,
)


def test_compute_regression_metrics_perfect():
    """Verify metrics for perfect predictions."""
    y = np.array([100.0, 80.0, 50.0, 20.0, 0.0])
    metrics = compute_regression_metrics(y, y)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_compute_regression_metrics_known_diff():
    """Verify metrics calculation against hand-computed values."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])  # errors: +2, -2, 0. MAE = 4/3 = 1.3333, MSE = 8/3, RMSE = sqrt(8/3) ~= 1.6330
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(1.3333, abs=1e-4)
    assert metrics["rmse"] == pytest.approx(1.6330, abs=1e-4)


def test_compute_regression_metrics_invalid_inputs():
    """Verify error handling on empty or mismatched input arrays."""
    with pytest.raises(ValueError, match="must not be empty"):
        compute_regression_metrics([], [])

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_regression_metrics([1, 2], [1])


def test_calculate_early_warning_lead_time_with_anomaly():
    """Verify lead time calculation when anomaly is detected."""
    df = pd.DataFrame({
        "cycle": [10, 20, 30, 40, 50],
        "Anomaly_Score": [0.2, 0.1, -0.05, -0.10, -0.25],
    })
    res = calculate_early_warning_lead_time(df, threshold=0.0)

    assert res["is_anomalous"] is True
    assert res["first_anomaly_cycle"] == 30
    assert res["max_cycle"] == 50
    assert res["lead_time"] == 20  # 50 - 30
    assert res["anomaly_count"] == 3


def test_calculate_early_warning_lead_time_no_anomaly():
    """Verify lead time is 0 when all scores remain normal."""
    df = pd.DataFrame({
        "cycle": [1, 2, 3],
        "Anomaly_Score": [0.15, 0.22, 0.35],
    })
    res = calculate_early_warning_lead_time(df, threshold=0.0)

    assert res["is_anomalous"] is False
    assert res["first_anomaly_cycle"] is None
    assert res["max_cycle"] == 3
    assert res["lead_time"] == 0
    assert res["anomaly_count"] == 0


def test_calculate_early_warning_lead_time_empty():
    """Verify graceful empty dataframe handling."""
    res = calculate_early_warning_lead_time(pd.DataFrame())
    assert res["lead_time"] == 0
    assert res["first_anomaly_cycle"] is None
    assert res["is_anomalous"] is False


def test_get_anomaly_summary():
    """Verify summary dictionary aggregation and health status tagging."""
    df_normal = pd.DataFrame({
        "cycle": [1, 2],
        "Anomaly_Score": [0.1, 0.05],
    })
    summary_norm = get_anomaly_summary(df_normal, threshold=0.0)
    assert summary_norm["status"] == "NORMAL"
    assert summary_norm["latest_score"] == 0.05

    df_critical = pd.DataFrame({
        "cycle": [1, 2],
        "Anomaly_Score": [0.1, -0.15],
    })
    summary_crit = get_anomaly_summary(df_critical, threshold=0.0)
    assert summary_crit["status"] == "HIGH RISK"
    assert summary_crit["latest_score"] == -0.15
