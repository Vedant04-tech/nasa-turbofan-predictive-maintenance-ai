import os
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent

def get_sample_data_path() -> Path:
    p = BASE_DIR / "engine_samples.csv"
    if p.exists():
        return p
    return BASE_DIR / "#engine_samples.csv"

def test_models_exist_and_loadable():
    """Verify all model and feature artifacts are present and can be deserialized."""
    rul_model_path = BASE_DIR / "rul_model.pkl"
    anomaly_model_path = BASE_DIR / "anomaly_model.pkl"
    feature_cols_path = BASE_DIR / "feature_cols.pkl"

    assert rul_model_path.exists(), "rul_model.pkl is missing"
    assert anomaly_model_path.exists(), "anomaly_model.pkl is missing"
    assert feature_cols_path.exists(), "feature_cols.pkl is missing"

    rf = joblib.load(rul_model_path)
    iso = joblib.load(anomaly_model_path)
    feature_cols = joblib.load(feature_cols_path)

    assert rf is not None
    assert iso is not None
    assert isinstance(feature_cols, list)
    assert len(feature_cols) > 0

def test_dataset_and_feature_columns_match():
    """Verify the sample dataset contains all required feature columns."""
    data_path = get_sample_data_path()
    assert data_path.exists(), f"Sample data file is missing: {data_path}"

    feature_cols = joblib.load(BASE_DIR / "feature_cols.pkl")
    df = pd.read_csv(data_path)

    missing = [col for col in feature_cols if col not in df.columns]
    assert not missing, f"Dataset missing expected feature columns: {missing}"

def test_threshold_reactivity_logic():
    """
    Test that anomaly detection and lead time calculation are reactive to
    the sensitivity threshold.
    """
    data_path = get_sample_data_path()
    feature_cols = joblib.load(BASE_DIR / "feature_cols.pkl")
    iso = joblib.load(BASE_DIR / "anomaly_model.pkl")

    df = pd.read_csv(data_path)
    engine_df = df[df["id"] == 2].copy().sort_values("cycle")
    X = engine_df[feature_cols]

    engine_df["Anomaly_Score"] = iso.decision_function(X)

    # If the logic correctly couples Anomaly_Flag to threshold:
    # At threshold 0.0:
    thresh_default = 0.0
    flags_default = np.where(engine_df["Anomaly_Score"] < thresh_default, -1, 1)
    anom_points_default = engine_df[flags_default == -1]

    # At threshold -0.05:
    thresh_tuned = -0.05
    flags_tuned = np.where(engine_df["Anomaly_Score"] < thresh_tuned, -1, 1)
    anom_points_tuned = engine_df[flags_tuned == -1]

    failure_cycle = engine_df["cycle"].max()

    lead_time_default = failure_cycle - anom_points_default["cycle"].min()
    lead_time_tuned = failure_cycle - anom_points_tuned["cycle"].min()

    # The tuned threshold should filter out early false-positive noise
    assert len(anom_points_tuned) < len(anom_points_default), "Tuned threshold should reduce anomaly count"
    assert lead_time_tuned < lead_time_default, "Tuned threshold should eliminate spurious early lead time"
    assert lead_time_tuned == 46, f"Expected 46 cycles lead time at -0.05 threshold, got {lead_time_tuned}"

def test_app_test_slider_reactivity():
    """
    Test using Streamlit AppTest that moving the sensitivity slider updates
    both the anomaly events table and the lead time display.
    """
    from streamlit.testing.v1 import AppTest

    data_path = get_sample_data_path()
    with open(data_path, "rb") as f:
        content = f.read()

    at = AppTest.from_file("Ai_Maintainence_Predictor.py", default_timeout=30)
    at.run(timeout=30)
    assert not at.exception, f"App threw exception on start: {at.exception}"

    # Upload file
    at.sidebar.file_uploader[0].upload("engine_samples.csv", content)
    at.run(timeout=30)
    assert not at.exception, f"App threw exception after upload: {at.exception}"

    # Select engine 2
    at.sidebar.selectbox[0].select(2)
    at.run(timeout=30)

    # Default slider is 0.0
    initial_rows = len(at.dataframe[0].value) if at.dataframe else 0

    # Extract initial lead time from markdown
    initial_lead_time_md = [m.value for m in at.markdown if "Early Warning Lead Time" in m.value]
    assert initial_lead_time_md, "Early Warning Lead Time metric card not found"

    # Set slider to -0.05
    at.sidebar.slider[0].set_value(-0.05)
    at.run(timeout=30)
    assert not at.exception, f"App threw exception after slider change: {at.exception}"

    tuned_rows = len(at.dataframe[0].value) if at.dataframe else 0
    tuned_lead_time_md = [m.value for m in at.markdown if "Early Warning Lead Time" in m.value]

    # Verify that the table and lead time changed in response to the slider
    assert tuned_rows != initial_rows, (
        f"Dataframe rows did not change when slider changed: {initial_rows} vs {tuned_rows}"
    )
    assert "46 cycles" in tuned_lead_time_md[0], (
        f"Expected '46 cycles' in tuned lead time metric, got: {tuned_lead_time_md[0]}"
    )
