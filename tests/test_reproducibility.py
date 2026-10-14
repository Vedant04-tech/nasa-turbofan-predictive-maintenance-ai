"""
Unit and integration tests for reproducible evaluation pipeline, RUL target formulation,
engine-level split integrity, and anomaly detection performance metrics.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from turbofan_ai.data import add_piecewise_rul_target
from turbofan_ai.evaluation import (
    compute_regression_metrics,
    evaluate_anomaly_detection_performance,
)


def test_piecewise_rul_target_exact():
    """Verify piece-wise linear RUL formulation caps at plateau and decreases linearly to 0."""
    df = pd.DataFrame({
        "id": [1] * 10,
        "cycle": list(range(1, 11)),
    })
    # Terminal cycle is 10. Uncapped RUL would be: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    capped_df = add_piecewise_rul_target(df, rul_cap=5)
    expected_rul = [5, 5, 5, 5, 5, 4, 3, 2, 1, 0]
    assert list(capped_df["RUL"]) == expected_rul

    # Terminal cycle must be strictly 0
    assert capped_df.loc[9, "RUL"] == 0

    # Test uncapped (rul_cap=None)
    uncapped_df = add_piecewise_rul_target(df, rul_cap=None)
    assert list(uncapped_df["RUL"]) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    # Invalid non-positive cap
    with pytest.raises(ValueError, match="positive integer"):
        add_piecewise_rul_target(df, rul_cap=0)

    # Missing columns
    with pytest.raises(ValueError, match="must contain 'id' and 'cycle'"):
        add_piecewise_rul_target(pd.DataFrame({"id": [1, 2]}))


def test_engine_level_split_zero_leakage():
    """Verify engine-level split guarantees strictly disjoint engine IDs between train and test."""
    from scripts.evaluate import prepare_engine_split

    # Create synthetic multi-engine dataset with 10 engines, 20 cycles each
    records = []
    for eng in range(1, 11):
        for c in range(1, 21):
            row = {"id": eng, "cycle": c, "op_set_1": 0.0, "op_set_2": 0.0, "op_set_3": 100.0}
            for s in range(1, 22):
                row[f"sensor_{s}"] = float(100 + s + c)
            records.append(row)
    raw_df = pd.DataFrame(records)

    train_df, test_df, train_ids, test_ids = prepare_engine_split(
        raw_df, rul_cap=130, test_size=0.3, random_state=42
    )

    # Strict set disjointness check
    assert len(train_ids) == 7
    assert len(test_ids) == 3
    assert set(train_ids).isdisjoint(set(test_ids)), "Zero engine leakage constraint violated!"

    # Every cycle in test_df belongs strictly to test_ids
    assert set(test_df["id"].unique()) == set(test_ids)
    assert set(train_df["id"].unique()) == set(train_ids)

    # Check chronological ordering
    for eng in test_ids:
        eng_sub = test_df[test_df["id"] == eng]
        assert eng_sub["cycle"].is_monotonic_increasing


def test_regression_metrics_mathematical_precision():
    """Verify regression metrics calculate exact RMSE, MAE, and R^2 on deterministic inputs."""
    y_true = np.array([100.0, 50.0, 25.0, 0.0])
    y_pred = np.array([110.0, 45.0, 25.0, 5.0])
    # Errors: [+10, -5, 0, +5].
    # MAE = (10 + 5 + 0 + 5) / 4 = 5.0
    # MSE = (100 + 25 + 0 + 25) / 4 = 37.5. RMSE = sqrt(37.5) = 6.1237
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == 5.0
    assert metrics["rmse"] == pytest.approx(6.1237, abs=1e-4)
    assert metrics["r2"] > 0.95

    # Perfect prediction must return exactly 0 error and R^2 = 1.0
    perf = compute_regression_metrics(y_true, y_true)
    assert perf["rmse"] == 0.0
    assert perf["mae"] == 0.0
    assert perf["r2"] == 1.0


def test_leakage_sensitive_scaler_isolation():
    """Verify learned preprocessing statistics fitted on training engines do not leak test statistics."""
    # Training engine distribution: mean 10.0
    X_train = pd.DataFrame({"sensor_11": np.array([8.0, 10.0, 12.0])})
    # Test engine distribution: shifted mean 100.0
    X_test = pd.DataFrame({"sensor_11": np.array([90.0, 100.0, 110.0])})

    scaler = StandardScaler()
    scaler.fit(X_train)

    # Scaler mean must strictly equal training mean
    assert scaler.mean_[0] == 10.0

    # Transforming test must use training mean, not test mean
    X_test_scaled = scaler.transform(X_test)
    # If using train mean (10), 100.0 scaled should be (100 - 10) / std != 0.0
    assert X_test_scaled[1, 0] > 0.0


def test_evaluate_anomaly_detection_performance_regimes():
    """Verify anomaly detection metrics accurately separate nominal false alarms from degraded detections."""
    # 1 engine with 10 cycles. RUL: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    # Let nominal cutoff be RUL > 5 (cycles 1-4, count=4).
    # Let degraded cutoff be RUL <= 2 (cycles 8-10, count=3).
    # Cycle 2 is a false alarm (< 0.0). Cycles 9 and 10 are degraded detections.
    df = pd.DataFrame({
        "id": [1] * 10,
        "cycle": list(range(1, 11)),
        "RUL": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        "Anomaly_Score": [0.2, -0.1, 0.3, 0.4, 0.1, 0.2, 0.05, 0.1, -0.2, -0.3],
    })

    results = evaluate_anomaly_detection_performance(
        df,
        threshold=0.0,
        nominal_rul_cutoff=5,
        degraded_rul_cutoff=2,
    )

    # Nominal cycles (RUL > 5): cycles 1, 2, 3, 4 (RUL 9, 8, 7, 6) -> 4 cycles
    assert results["nominal_cycles_evaluated"] == 4
    # Cycle 2 has score -0.1 < 0 -> 1 false alarm
    assert results["nominal_false_alarms"] == 1
    assert results["false_alarm_rate"] == 0.25  # 1 / 4

    # Degraded cycles (RUL <= 2): cycles 8, 9, 10 (RUL 2, 1, 0) -> 3 cycles
    assert results["degraded_cycles_evaluated"] == 3
    # Cycles 9 and 10 have scores -0.2 and -0.3 -> 2 detections
    assert results["degraded_detections"] == 2
    assert results["degraded_recall"] == pytest.approx(0.6667, abs=1e-4)

    # First trigger lead time: max_cycle is 10, first anomaly at cycle 2 -> lead = 8
    assert results["first_trigger_lead_time"]["mean"] == 8.0


def test_evaluate_anomaly_detection_performance_invalid_input():
    """Verify error handling on missing columns or empty input in anomaly evaluation."""
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_anomaly_detection_performance(pd.DataFrame())

    df_valid = pd.DataFrame({
        "id": [1],
        "cycle": [1],
        "RUL": [10],
        "Anomaly_Score": [0.1],
    })

    # Missing RUL column
    with pytest.raises(KeyError, match="Target column 'RUL' not found"):
        evaluate_anomaly_detection_performance(df_valid.drop(columns=["RUL"]))

    # Missing anomaly score column
    with pytest.raises(KeyError, match="Column 'custom_score' not found"):
        evaluate_anomaly_detection_performance(df_valid, score_col="custom_score")


def test_evaluate_regressors_pipeline_mock():
    """Verify evaluate_regressors pipeline handles baseline and model evaluation consistently."""
    from scripts.evaluate import evaluate_regressors

    feature_cols = ["sensor_11_rolling_mean", "sensor_4"]
    train_df = pd.DataFrame({
        "id": [1, 1],
        "cycle": [1, 2],
        "RUL": [100.0, 99.0],
        "sensor_11_rolling_mean": [500.0, 501.0],
        "sensor_4": [1400.0, 1401.0],
    })
    test_df = pd.DataFrame({
        "id": [2, 2],
        "cycle": [1, 2],
        "RUL": [50.0, 49.0],
        "sensor_11_rolling_mean": [510.0, 511.0],
        "sensor_4": [1410.0, 1411.0],
    })

    mock_rf = MagicMock()
    mock_rf.predict.return_value = np.array([51.0, 48.0])
    mock_rf.n_estimators = 200
    mock_rf.random_state = 42

    results, y_test, y_pred_rf, y_pred_lr = evaluate_regressors(
        train_df, test_df, feature_cols, mock_rf
    )

    assert "naive_mean_baseline" in results
    assert "linear_regression_baseline" in results
    assert "random_forest_regressor" in results

    assert results["random_forest_regressor"]["rmse"] == pytest.approx(1.0)
    assert len(y_test) == 2
    assert len(y_pred_rf) == 2
