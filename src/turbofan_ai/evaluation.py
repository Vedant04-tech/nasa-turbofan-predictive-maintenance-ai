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
) -> Dict[str, Any]:
    """
    Calculates the early warning lead time relative to the maximum recorded cycle.

    Important Semantic Note:
    -----------------------
    In run-to-failure datasets (such as NASA C-MAPSS FD001 training data), the maximum
    recorded cycle corresponds to the actual functional breakdown point. In in-service
    or right-censored industrial streaming data where failure has not yet occurred, this
    metric represents the elapsed monitoring interval between the initial warning trigger
    and the latest recorded cycle, not a confirmed time-to-failure estimate.

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

    Returns
    -------
    Dict[str, Any]
        {
            'lead_time': int,
            'first_anomaly_cycle': Optional[int],
            'max_cycle': int,
            'anomaly_count': int,
            'is_anomalous': bool
        }
    """
    if engine_df is None or engine_df.empty:
        return {
            "lead_time": 0,
            "first_anomaly_cycle": None,
            "max_cycle": 0,
            "anomaly_count": 0,
            "is_anomalous": False,
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

    if not anomaly_points.empty:
        first_anomaly = int(anomaly_points[cycle_col].min())
        lead_time = max(0, max_cycle - first_anomaly)
        return {
            "lead_time": lead_time,
            "first_anomaly_cycle": first_anomaly,
            "max_cycle": max_cycle,
            "anomaly_count": anomaly_count,
            "is_anomalous": True,
        }
    else:
        return {
            "lead_time": 0,
            "first_anomaly_cycle": None,
            "max_cycle": max_cycle,
            "anomaly_count": 0,
            "is_anomalous": False,
        }


def get_anomaly_summary(
    engine_df: pd.DataFrame,
    threshold: float = 0.0,
    score_col: str = "Anomaly_Score",
) -> Dict[str, Any]:
    """
    Provides a comprehensive anomaly health summary for an engine.
    """
    lead_info = calculate_early_warning_lead_time(
        engine_df, threshold=threshold, score_col=score_col
    )
    actual_score_col = score_col if score_col in engine_df.columns else "anomaly_score"
    scores = engine_df[actual_score_col].to_numpy() if actual_score_col in engine_df.columns else np.array([0.0])

    latest_score = float(scores[-1]) if len(scores) > 0 else 0.0
    min_score = float(np.min(scores)) if len(scores) > 0 else 0.0

    return {
        **lead_info,
        "latest_score": round(latest_score, 4),
        "min_score": round(min_score, 4),
        "status": "HIGH RISK" if latest_score < threshold else "NORMAL",
    }
