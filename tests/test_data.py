"""
Unit tests for turbofan_ai.data (validation, cleaning, and causal rolling feature generation).
"""

import numpy as np
import pandas as pd
import pytest

from turbofan_ai.data import (
    compute_rolling_features,
    validate_and_prepare_data,
)


def test_compute_rolling_features_causality():
    """Ensure rolling mean and std are purely causal and grouped by engine ID."""
    df = pd.DataFrame({
        "id": [1, 1, 1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 4, 5, 1, 2],
        "sensor_11": [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0],
    })

    result = compute_rolling_features(df, window=3, sensors=["sensor_11"])

    assert "sensor_11_rolling_mean" in result.columns
    assert "sensor_11_rolling_std" in result.columns

    # For engine 1 at cycle 1: mean is 10.0, std is 0.0 (imputed for single point)
    assert result.loc[0, "sensor_11_rolling_mean"] == 10.0
    assert result.loc[0, "sensor_11_rolling_std"] == 0.0

    # For engine 1 at cycle 3: mean is (10 + 20 + 30) / 3 = 20.0
    assert result.loc[2, "sensor_11_rolling_mean"] == pytest.approx(20.0)

    # For engine 1 at cycle 5 with window 3: mean is (30 + 40 + 50) / 3 = 40.0
    assert result.loc[4, "sensor_11_rolling_mean"] == pytest.approx(40.0)

    # For engine 2 at cycle 1: MUST NOT leak from engine 1
    assert result.loc[5, "sensor_11_rolling_mean"] == 100.0
    assert result.loc[5, "sensor_11_rolling_std"] == 0.0


def test_validate_and_prepare_data_valid_sample(sample_dataframe):
    """Verify validation passes on canonical sample dataset."""
    prepared, warnings = validate_and_prepare_data(sample_dataframe)
    assert not prepared.empty
    assert len(prepared) == len(sample_dataframe)
    assert "id" in prepared.columns
    assert "cycle" in prepared.columns


def test_validate_and_prepare_data_auto_rolling_derivation(synthetic_raw_engine_data):
    """Verify missing rolling features are automatically calculated from raw sensor columns."""
    expected_features = [
        "op_set_1",
        "op_set_2",
        "sensor_2",
        "sensor_11_rolling_mean",
        "sensor_11_rolling_std",
    ]
    prepared, warnings = validate_and_prepare_data(
        synthetic_raw_engine_data, feature_cols=expected_features
    )
    assert "sensor_11_rolling_mean" in prepared.columns
    assert "sensor_11_rolling_std" in prepared.columns
    assert any("Automatically computed causal rolling features" in w for w in warnings)


def test_validate_and_prepare_data_empty():
    """Verify empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        validate_and_prepare_data(pd.DataFrame())

    with pytest.raises(ValueError, match="empty"):
        validate_and_prepare_data(None)


def test_validate_and_prepare_data_missing_metadata():
    """Verify missing 'id' or 'cycle' raises ValueError."""
    df_missing_id = pd.DataFrame({"cycle": [1, 2], "sensor_1": [10.0, 11.0]})
    with pytest.raises(ValueError, match="Missing required identifier column 'id'"):
        validate_and_prepare_data(df_missing_id)

    df_missing_cycle = pd.DataFrame({"id": [1, 1], "sensor_1": [10.0, 11.0]})
    with pytest.raises(ValueError, match="Missing required identifier column 'cycle'"):
        validate_and_prepare_data(df_missing_cycle)


def test_validate_and_prepare_data_non_numeric_metadata():
    """Verify corrupt non-numeric metadata raises ValueError."""
    df_corrupt = pd.DataFrame({
        "id": ["eng_A", "eng_B"],
        "cycle": [1, 2],
    })
    with pytest.raises(ValueError, match="Column 'id' contains non-numeric"):
        validate_and_prepare_data(df_corrupt)


def test_validate_and_prepare_data_missing_features_unsolvable():
    """Verify missing model features that cannot be derived raise schema mismatch error."""
    df = pd.DataFrame({
        "id": [1, 1],
        "cycle": [1, 2],
        "sensor_2": [10.0, 12.0],
    })
    expected = ["sensor_2", "exotic_sensor_99"]
    with pytest.raises(ValueError, match="Dataset schema mismatch"):
        validate_and_prepare_data(df, feature_cols=expected)


def test_validate_and_prepare_data_inf_and_nan_rejection():
    """Verify infinite and NaN values in base features are strictly rejected rather than silently corrupted."""
    df_inf = pd.DataFrame({
        "id": [1, 1],
        "cycle": [1, 2],
        "sensor_2": [10.0, np.inf],
    })
    with pytest.raises(ValueError, match="infinite values"):
        validate_and_prepare_data(df_inf, feature_cols=["sensor_2"])

    df_nan = pd.DataFrame({
        "id": [1, 1],
        "cycle": [1, 2],
        "sensor_2": [10.0, np.nan],
    })
    with pytest.raises(ValueError, match="missing or NaN values"):
        validate_and_prepare_data(df_nan, feature_cols=["sensor_2"])


def test_validate_and_prepare_data_cycle_1_rolling_std_handling():
    """Verify legitimate cycle-1 rolling std NaN is imputed with 0.0, while post-cycle-1 NaNs fail."""
    # Legitimate cycle 1 rolling std NaN
    df_cycle_1 = pd.DataFrame({
        "id": [1, 1],
        "cycle": [1, 2],
        "sensor_11_rolling_std": [np.nan, 2.5],
    })
    prepared, _ = validate_and_prepare_data(
        df_cycle_1, feature_cols=["sensor_11_rolling_std"]
    )
    assert prepared.loc[0, "sensor_11_rolling_std"] == 0.0
    assert prepared.loc[1, "sensor_11_rolling_std"] == 2.5

    # Illegitimate post-cycle-1 NaN
    df_corrupt_std = pd.DataFrame({
        "id": [1, 1],
        "cycle": [1, 2],
        "sensor_11_rolling_std": [0.0, np.nan],
    })
    with pytest.raises(ValueError, match="beyond initial cycle"):
        validate_and_prepare_data(
            df_corrupt_std, feature_cols=["sensor_11_rolling_std"]
        )


def test_validate_and_prepare_data_duplicates_and_invalid_cycles():
    """Verify duplicate engine cycles and invalid cycle numbers are caught."""
    # Duplicate record
    df_dup = pd.DataFrame({
        "id": [1, 1, 1],
        "cycle": [1, 2, 2],
        "sensor_2": [10.0, 11.0, 12.0],
    })
    with pytest.raises(ValueError, match="duplicate record"):
        validate_and_prepare_data(df_dup, feature_cols=["sensor_2"])

    # Non-positive cycle
    df_invalid_cycle = pd.DataFrame({
        "id": [1, 1],
        "cycle": [0, 1],
        "sensor_2": [10.0, 11.0],
    })
    with pytest.raises(ValueError, match="non-positive cycle"):
        validate_and_prepare_data(df_invalid_cycle, feature_cols=["sensor_2"])

    # Non-finite cycle
    df_inf_cycle = pd.DataFrame({
        "id": [1, 1],
        "cycle": [1, np.inf],
        "sensor_2": [10.0, 11.0],
    })
    with pytest.raises(ValueError, match="non-finite"):
        validate_and_prepare_data(df_inf_cycle, feature_cols=["sensor_2"])
