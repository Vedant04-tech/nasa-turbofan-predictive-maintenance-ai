"""
Data validation, cleaning, and causal feature engineering for turbofan sensor telemetry.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

RAW_METADATA_COLS = ["id", "cycle"]
OP_SET_COLS = ["op_set_1", "op_set_2", "op_set_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_RAW_COLS = RAW_METADATA_COLS + OP_SET_COLS + SENSOR_COLS

# Sensors eliminated during exploratory data analysis due to near-zero variance
LOW_VARIANCE_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

# Sensors exhibiting strong monotonic correlation with degradation
ROLLING_SENSORS = [
    "sensor_11",
    "sensor_4",
    "sensor_12",
    "sensor_7",
    "sensor_15",
    "sensor_21",
]

ROLLING_WINDOW = 5


def compute_rolling_features(
    df: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    sensors: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes causal (backward-looking) rolling mean and standard deviation grouped by engine ID.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing engine telemetry, sorted by 'id' and 'cycle'.
    window : int
        Rolling window size (default: 5).
    sensors : list of str, optional
        Sensors on which to calculate rolling statistics (default: ROLLING_SENSORS).

    Returns
    -------
    pd.DataFrame
        DataFrame with added `<sensor>_rolling_mean` and `<sensor>_rolling_std` columns.
    """
    if sensors is None:
        sensors = ROLLING_SENSORS

    df = df.copy()

    # Ensure sequential cycle ordering within each engine to guarantee causal rolling windows
    if "id" in df.columns and "cycle" in df.columns:
        df = df.sort_values(by=["id", "cycle"]).reset_index(drop=True)

    for sensor in sensors:
        if sensor in df.columns:
            mean_col = f"{sensor}_rolling_mean"
            std_col = f"{sensor}_rolling_std"

            # min_periods=1 ensures predictions can be made starting from cycle 1
            if "id" in df.columns:
                grouped = df.groupby("id")[sensor]
                df[mean_col] = grouped.transform(
                    lambda s: s.rolling(window, min_periods=1).mean()
                )
                df[std_col] = grouped.transform(
                    lambda s: s.rolling(window, min_periods=1).std()
                )
            else:
                df[mean_col] = df[sensor].rolling(window, min_periods=1).mean()
                df[std_col] = df[sensor].rolling(window, min_periods=1).std()

            # Impute rolling std at single-sample points (cycle=1) where degrees of freedom = 0 (std is NaN)
            df[std_col] = df[std_col].fillna(0.0)

    return df


def add_piecewise_rul_target(
    df: pd.DataFrame,
    rul_cap: Optional[int] = 130,
) -> pd.DataFrame:
    """
    Computes piece-wise linear Remaining Useful Life (RUL) target for run-to-failure engines.

    In NASA C-MAPSS FD001, initial degradation is undetectable. The RUL is therefore
    formulated with a constant plateau during early healthy operating cycles, capped at
    a threshold (e.g. 130 cycles), and decreases linearly toward 0 at failure:
        RUL(t) = min(max_cycle - t, rul_cap)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing engine telemetry with 'id' and 'cycle' columns.
    rul_cap : int, optional
        Maximum RUL threshold for piece-wise degradation clipping (default: 130).
        If None, uncapped linear RUL is computed.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'RUL' column.

    Raises
    ------
    ValueError
        If 'id' or 'cycle' columns are missing or if rul_cap is non-positive.
    """
    if "id" not in df.columns or "cycle" not in df.columns:
        raise ValueError("Dataset must contain 'id' and 'cycle' columns to compute RUL.")

    if rul_cap is not None and rul_cap <= 0:
        raise ValueError(f"rul_cap must be a positive integer, got {rul_cap}")

    df = df.copy()
    max_cycles = df.groupby("id")["cycle"].transform("max")
    rul = max_cycles - df["cycle"]

    if rul_cap is not None:
        rul = np.minimum(rul, rul_cap)

    df["RUL"] = rul
    return df


def validate_and_prepare_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validates uploaded engine telemetry data, derives missing rolling features if raw
    sensors are provided, checks numeric integrity, and aligns columns for inference.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe from CSV upload.
    feature_cols : list of str, optional
        Expected list and order of feature columns required by trained models.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        (prepared_dataframe, list_of_warning_messages)

    Raises
    ------
    ValueError
        If dataset is empty, missing critical metadata columns, or cannot satisfy feature schema.
    """
    warnings: List[str] = []

    if df is None or df.empty:
        raise ValueError("The provided dataset is empty.")

    prepared = df.copy()

    # Check for required engine identifier and cycle metadata, and validate numeric types
    for req in RAW_METADATA_COLS:
        if req not in prepared.columns:
            raise ValueError(
                f"Missing required identifier column '{req}'. Dataset must contain 'id' and 'cycle'."
            )
        if not pd.api.types.is_numeric_dtype(prepared[req]):
            try:
                prepared[req] = pd.to_numeric(prepared[req], errors="coerce")
            except Exception as e:
                raise ValueError(
                    f"Column '{req}' could not be converted to numeric values: {e}"
                )
        if not np.isfinite(prepared[req].to_numpy()).all():
            raise ValueError(
                f"Column '{req}' contains non-numeric, non-finite, or null values."
            )

    # Validate valid cycle values (operational cycles must be >= 1)
    if (prepared["cycle"] < 1).any():
        raise ValueError("Column 'cycle' contains invalid non-positive cycle values.")

    # Detect duplicate cycle records per engine
    duplicates = prepared.duplicated(subset=["id", "cycle"])
    if duplicates.any():
        dup_count = int(duplicates.sum())
        raise ValueError(
            f"Dataset contains {dup_count} duplicate record(s) for the same engine ID and cycle."
        )

    # Sort cycles chronologically per engine
    prepared = prepared.sort_values(by=["id", "cycle"]).reset_index(drop=True)

    # If raw NASA space-separated columns or sensor columns exist without rolling features, compute them
    if feature_cols is not None:
        missing_features = [c for c in feature_cols if c not in prepared.columns]
        rolling_missing = [c for c in missing_features if "rolling" in c]

        if rolling_missing:
            # Check if underlying base sensors are present to derive rolling features
            available_sensors = [s for s in ROLLING_SENSORS if s in prepared.columns]
            if available_sensors:
                prepared = compute_rolling_features(prepared, sensors=available_sensors)
                warnings.append(
                    f"Automatically computed causal rolling features for sensors: {', '.join(available_sensors)}."
                )

        # Re-check for any still-missing features
        still_missing = [c for c in feature_cols if c not in prepared.columns]
        if still_missing:
            raise ValueError(
                f"Dataset schema mismatch: missing required model features: {still_missing}"
            )

        # Validate numeric types and finite values in feature matrix
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(prepared[col]):
                try:
                    prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
                except Exception:
                    raise ValueError(
                        f"Feature '{col}' contains non-numeric values."
                    )

            # Check for infinite values (unphysical sensor readings)
            if np.isinf(prepared[col].to_numpy()).any():
                raise ValueError(
                    f"Feature '{col}' contains infinite values."
                )

            # Check for missing/null values without silent corruption of base sensors
            if prepared[col].isna().any():
                if col.endswith("_rolling_std"):
                    # Legitimate mathematical edge case: cycle=1 single-sample std is NaN (zero variance)
                    cycle_1_mask = prepared["cycle"] == 1
                    nan_mask = prepared[col].isna()
                    if (nan_mask & ~cycle_1_mask).any():
                        raise ValueError(
                            f"Feature '{col}' contains missing or NaN values beyond initial cycle."
                        )
                    prepared.loc[nan_mask, col] = 0.0
                else:
                    raise ValueError(
                        f"Feature '{col}' contains missing or NaN values."
                    )

    return prepared, warnings
