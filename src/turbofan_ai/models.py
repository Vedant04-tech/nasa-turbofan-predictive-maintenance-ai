"""
Model artifact loading, inference pipelines, and scoring wrappers.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd


def get_default_model_dir() -> Path:
    """Finds the model directory across installed package or project workspace."""
    # Priority 1: Project root models/ directory relative to this file (src/turbofan_ai/models.py)
    candidate_1 = Path(__file__).resolve().parent.parent.parent / "models"
    if candidate_1.exists():
        return candidate_1

    # Priority 2: Working directory models/
    candidate_2 = Path.cwd() / "models"
    if candidate_2.exists():
        return candidate_2

    # Priority 3: Working directory root fallback
    return Path.cwd()


def load_model_artifacts(
    model_dir: Optional[Union[str, Path]] = None,
) -> Tuple[object, object, List[str]]:
    """
    Safely deserializes and validates the trained RUL and Anomaly Detection models.

    Parameters
    ----------
    model_dir : str or Path, optional
        Directory containing 'rul_model.pkl', 'anomaly_model.pkl', and 'feature_cols.pkl'.

    Returns
    -------
    Tuple[RandomForestRegressor, IsolationForest, List[str]]
        (rf_model, isolation_forest_model, feature_columns)

    Raises
    ------
    FileNotFoundError
        If any model artifact is missing.
    RuntimeError
        If deserialized models fail expected interface verification.
    """
    if model_dir is None:
        target_dir = get_default_model_dir()
    else:
        target_dir = Path(model_dir)

    rul_path = target_dir / "rul_model.pkl"
    iso_path = target_dir / "anomaly_model.pkl"
    features_path = target_dir / "feature_cols.pkl"

    for artifact in [rul_path, iso_path, features_path]:
        if not artifact.exists():
            raise FileNotFoundError(
                f"Required model artifact not found: '{artifact.resolve()}'. "
                f"Verify that Git LFS pulled model files or specify the correct model_dir."
            )

    rf_model = joblib.load(rul_path)
    iso_model = joblib.load(iso_path)
    feature_cols = joblib.load(features_path)

    if not hasattr(rf_model, "predict"):
        raise RuntimeError("Loaded RUL model artifact lacks a valid 'predict' method.")

    if not hasattr(iso_model, "decision_function"):
        raise RuntimeError(
            "Loaded Anomaly Detection model lacks a valid 'decision_function' method."
        )

    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise RuntimeError("Loaded feature_cols specification is empty or not a list.")

    return rf_model, iso_model, feature_cols


def predict_rul(rf_model: object, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Executes RUL regression inference with physical non-negativity constraint.

    Parameters
    ----------
    rf_model : object
        Trained regression model with `.predict()`.
    X : pd.DataFrame or np.ndarray
        Feature matrix matching training schema.

    Returns
    -------
    np.ndarray
        Array of predicted Remaining Useful Life (non-negative float/int).
    """
    preds = rf_model.predict(X)
    # Remaining useful life in operating machinery cannot physically be negative
    return np.maximum(0.0, preds)


def predict_anomaly(
    iso_model: object,
    X: Union[pd.DataFrame, np.ndarray],
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes anomaly decision scores and dynamic threshold-dependent anomaly flags.

    Parameters
    ----------
    iso_model : object
        Trained Isolation Forest model.
    X : pd.DataFrame or np.ndarray
        Feature matrix matching training schema.
    threshold : float, optional
        Decision boundary score below which observations are flagged abnormal (default: 0.0).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (anomaly_scores, anomaly_flags) where anomaly_flags is -1 for anomaly and +1 for normal.
    """
    scores = iso_model.decision_function(X)
    flags = np.where(scores < threshold, -1, 1)
    return scores, flags
