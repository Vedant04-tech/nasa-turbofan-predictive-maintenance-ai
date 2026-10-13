"""
Evaluation metrics, prognostic error calculations, and early warning lead-time semantics.
"""

from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Computes standard prognostic regression error metrics: RMSE, MAE, and R^2.

    Parameters
    ----------
    y_true : array-like
        Ground truth target Remaining Useful Life values.
    y_pred : array-like
        Predicted Remaining Useful Life values.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys 'rmse', 'mae', 'r2'.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) == 0 or len(y_p) == 0:
        raise ValueError("Inputs y_true and y_pred must not be empty.")

    if len(y_t) != len(y_p):
        raise ValueError("Length mismatch between y_true and y_pred.")

    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p)) if len(y_t) > 1 else 0.0

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
    }


def calculate_early_warning_lead_time(
    engine_df: pd.DataFrame,
    threshold: float = 0.0,
    score_col: str = "Anomaly_Score",
    cycle_col: str = "cycle",
    is_run_to_failure: bool = False,
) -> Dict[str, Any]:
    """
    Calculates the warning window relative to the maximum observed cycle.

    Important Prognostic Distinction:
    ---------------------------------
    In run-to-failure benchmarks (such as NASA C-MAPSS FD001), the terminal cycle represents
    an actual functional failure, making this interval a true failure lead time.
    For in-service or right-censored operating telemetry where failure has not occurred,
    the true failure cycle is unknown. In that case, this metric represents an observed
    detection window since the initial anomaly, not a confirmed time-to-failure estimate.

    Parameters
    ----------
    engine_df : pd.DataFrame
        Telemetry observations for a single engine ID.
    threshold : float, optional
        Anomaly sensitivity cutoff (default: 0.0). Scores below threshold are anomalies.
    score_col : str, optional
        Name of the anomaly score column (default: 'Anomaly_Score').
    cycle_col : str, optional
        Name of the cycle sequence column (default: 'cycle').
    is_run_to_failure : bool, optional
        Whether the telemetry represents a confirmed run-to-failure sequence (default: False).

    Returns
    -------
    Dict[str, Any]
        {
            'lead_time': int,
            'observed_interval': int,
            'first_anomaly_cycle': Optional[int],
            'max_cycle': int,
            'anomaly_count': int,
            'is_anomalous': bool,
            'is_run_to_failure': bool,
        }
    """
    if engine_df is None or engine_df.empty:
        return {
            "lead_time": 0,
            "observed_interval": 0,
            "first_anomaly_cycle": None,
            "max_cycle": 0,
            "anomaly_count": 0,
            "is_anomalous": False,
            "is_run_to_failure": is_run_to_failure,
        }

    # Support case-insensitive score column
    actual_score_col = score_col
    if score_col not in engine_df.columns:
        lower_cols = {c.lower(): c for c in engine_df.columns}
        if score_col.lower() in lower_cols:
            actual_score_col = lower_cols[score_col.lower()]
        else:
            raise KeyError(f"Column '{score_col}' not found in engine DataFrame.")

    sorted_df = engine_df.sort_values(by=cycle_col).reset_index(drop=True)
    max_cycle = int(sorted_df[cycle_col].max())

    anomaly_points = sorted_df[sorted_df[actual_score_col] < threshold]
    anomaly_count = len(anomaly_points)

    has_anomaly = not anomaly_points.empty
    first_anomaly = int(anomaly_points[cycle_col].min()) if has_anomaly else None
    interval = max(0, max_cycle - first_anomaly) if first_anomaly is not None else 0

    return {
        "lead_time": interval,
        "observed_interval": interval,
        "first_anomaly_cycle": first_anomaly,
        "max_cycle": max_cycle,
        "anomaly_count": anomaly_count,
        "is_anomalous": has_anomaly,
        "is_run_to_failure": is_run_to_failure,
    }


def get_anomaly_summary(
    engine_df: pd.DataFrame,
    threshold: float = 0.0,
    score_col: str = "Anomaly_Score",
    is_run_to_failure: bool = False,
) -> Dict[str, Any]:
    """
    Provides a comprehensive anomaly health summary for an engine.
    """
    lead_info = calculate_early_warning_lead_time(
        engine_df, threshold=threshold, score_col=score_col, is_run_to_failure=is_run_to_failure
    )
    col = score_col if score_col in engine_df.columns else "anomaly_score"
    scores = engine_df[col].to_numpy() if col in engine_df.columns else np.array([0.0])

    latest_score = float(scores[-1]) if len(scores) > 0 else 0.0
    min_score = float(np.min(scores)) if len(scores) > 0 else 0.0

    return {
        **lead_info,
        "latest_score": round(latest_score, 4),
        "min_score": round(min_score, 4),
        "status": "HIGH RISK" if latest_score < threshold else "NORMAL",
    }
