"""
Evaluation metrics, prognostic error calculations, and early warning lead-time semantics.
"""

from typing import Any, Dict, List, Union

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


def evaluate_anomaly_detection_performance(
    df: pd.DataFrame,
    threshold: float = 0.0,
    score_col: str = "Anomaly_Score",
    cycle_col: str = "cycle",
    rul_col: str = "RUL",
    nominal_rul_cutoff: int = 100,
    degraded_rul_cutoff: int = 30,
) -> Dict[str, Any]:
    """
    Evaluates unsupervised anomaly detection performance independently against operational regimes.

    Because run-to-failure benchmarks lack discrete ground-truth labels for anomaly
    injection, this function calculates objective metrics across two defined operating regimes:
    1. Nominal Regime (RUL > nominal_rul_cutoff): Measures the False Alarm Rate (FAR), where
       scores below the threshold represent premature false alerts.
    2. Degraded Regime (RUL <= degraded_rul_cutoff): Measures Detection Recall, where scores
       below the threshold represent successful degradation detection prior to failure.
    3. Fleet-wide Lead Time Distribution: Tracks the first alert cycle and sustained alert cycles
       (3 consecutive cycles) per engine, reporting min, median, mean, max, and std.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation telemetry containing engine IDs, cycles, anomaly scores, and RUL.
    threshold : float, optional
        Decision boundary cutoff (default: 0.0). Scores < threshold are anomalies.
    score_col : str, optional
        Anomaly score column name (default: 'Anomaly_Score').
    cycle_col : str, optional
        Cycle column name (default: 'cycle').
    rul_col : str, optional
        RUL column name (default: 'RUL').
    nominal_rul_cutoff : int, optional
        Cutoff RUL above which cycles are considered nominal baseline (default: 100).
    degraded_rul_cutoff : int, optional
        Cutoff RUL below which cycles are in the critical degradation phase (default: 30).

    Returns
    -------
    Dict[str, Any]
        Dictionary of anomaly evaluation metrics, detection recall, and lead-time distributions.
    """
    if df is None or df.empty:
        raise ValueError("Evaluation DataFrame must not be empty.")

    actual_score_col = score_col
    if score_col not in df.columns:
        lower_cols = {c.lower(): c for c in df.columns}
        if score_col.lower() in lower_cols:
            actual_score_col = lower_cols[score_col.lower()]
        else:
            raise KeyError(f"Column '{score_col}' not found in DataFrame.")

    if rul_col not in df.columns:
        raise KeyError(f"Target column '{rul_col}' not found in DataFrame.")

    if cycle_col not in df.columns or "id" not in df.columns:
        raise KeyError(f"Columns 'id' and '{cycle_col}' are required for fleet evaluation.")

    eval_df = df.copy()
    eval_df["is_anomaly"] = eval_df[actual_score_col] < threshold

    # 1. Regime-based evaluation
    nominal_mask = eval_df[rul_col] > nominal_rul_cutoff
    degraded_mask = eval_df[rul_col] <= degraded_rul_cutoff

    nominal_total = int(nominal_mask.sum())
    nominal_false_alarms = int(eval_df.loc[nominal_mask, "is_anomaly"].sum())
    false_alarm_rate = (
        round(nominal_false_alarms / nominal_total, 4) if nominal_total > 0 else 0.0
    )

    degraded_total = int(degraded_mask.sum())
    degraded_detections = int(eval_df.loc[degraded_mask, "is_anomaly"].sum())
    degraded_recall = (
        round(degraded_detections / degraded_total, 4) if degraded_total > 0 else 0.0
    )

    # 2. Fleet lead-time distribution across engines
    first_trigger_leads: List[int] = []
    sustained_3_leads: List[int] = []

    for _, group in eval_df.groupby("id"):
        sorted_group = group.sort_values(cycle_col).reset_index(drop=True)
        max_c = int(sorted_group[cycle_col].max())

        # First trigger lead time
        anom_rows = sorted_group[sorted_group["is_anomaly"]]
        if not anom_rows.empty:
            first_c = int(anom_rows[cycle_col].min())
            first_trigger_leads.append(max(0, max_c - first_c))
        else:
            first_trigger_leads.append(0)

        # Sustained trigger (3 consecutive anomaly flags)
        anom_series = sorted_group["is_anomaly"].astype(int)
        rolling_3 = anom_series.rolling(3).sum()
        sustained_rows = sorted_group[rolling_3 >= 3]
        if not sustained_rows.empty:
            sustained_c = int(sustained_rows[cycle_col].iloc[0]) - 2
            sustained_3_leads.append(max(0, max_c - sustained_c))
        else:
            sustained_3_leads.append(0)

    def _summarize_leads(leads: List[int]) -> Dict[str, float]:
        if not leads:
            return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0, "std": 0.0}
        arr = np.array(leads, dtype=float)
        return {
            "mean": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "std": round(float(np.std(arr)), 2),
        }

    return {
        "threshold": threshold,
        "nominal_rul_cutoff": nominal_rul_cutoff,
        "degraded_rul_cutoff": degraded_rul_cutoff,
        "nominal_cycles_evaluated": nominal_total,
        "nominal_false_alarms": nominal_false_alarms,
        "false_alarm_rate": false_alarm_rate,
        "degraded_cycles_evaluated": degraded_total,
        "degraded_detections": degraded_detections,
        "degraded_recall": degraded_recall,
        "first_trigger_lead_time": _summarize_leads(first_trigger_leads),
        "sustained_3_lead_time": _summarize_leads(sustained_3_leads),
    }
