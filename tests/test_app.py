"""
Integration tests for Streamlit application (app.py) using streamlit.testing.v1.AppTest.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app_path() -> str:
    return str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_initialization_with_demo_dataset(app_path):
    """Verify application boots up cleanly, loads benchmark dataset by default, and renders KPIs."""
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run(timeout=30)

    assert not at.exception, f"App threw unhandled exception on boot: {at.exception}"

    # Verify key dashboard headers and subheaders are rendered
    all_texts = (
        [m.value for m in at.markdown]
        + [s.value for s in at.subheader]
        + [t.value for t in at.title]
    )
    combined_content = " ".join(all_texts)

    assert "NASA Turbofan Predictive Maintenance Control Center" in combined_content
    assert "Remaining Useful Life" in combined_content
    assert "Early Warning Lead Time" in combined_content
    assert "Subsystem Risk Meter" in combined_content or "Risk" in combined_content


def test_app_slider_reactivity_and_filtering(app_path):
    """Verify threshold slider dynamically controls anomaly detections and UI state."""
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run(timeout=30)
    assert not at.exception

    # Select engine unit 2 if present
    if at.sidebar.selectbox:
        at.sidebar.selectbox[0].select(2)
        at.run(timeout=30)
        assert not at.exception

    assert at.sidebar.slider, "Threshold slider not found in sidebar"

    # Set slider to very strict threshold (-0.40) where few/no anomalies are triggered
    at.sidebar.slider[0].set_value(-0.40)
    at.run(timeout=30)
    assert not at.exception

    text_strict = " ".join([m.value for m in at.markdown])
    assert "Cutoff Boundary: -0.40" in text_strict
    rows_strict = len(at.dataframe[0].value) if at.dataframe else 0

    # Set slider to sensitive threshold (+0.40) where many anomalies are triggered
    at.sidebar.slider[0].set_value(0.40)
    at.run(timeout=30)
    assert not at.exception

    text_sensitive = " ".join([m.value for m in at.markdown])
    assert "Cutoff Boundary: +0.40" in text_sensitive
    assert at.dataframe, "Expected detected anomalies table at sensitive threshold"
    rows_sensitive = len(at.dataframe[0].value)

    # Increasing sensitivity must strictly detect more anomalies on this non-trivial dataset
    assert rows_sensitive > rows_strict, (
        f"Increasing threshold sensitivity should increase flagged anomaly rows: "
        f"{rows_strict} vs {rows_sensitive}"
    )


def test_app_handles_invalid_csv_gracefully(app_path):
    """Verify invalid CSV upload triggers graceful st.error instead of crashing with unhandled exception."""
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run(timeout=30)
    assert not at.exception

    # Switch radio to "Upload Custom CSV"
    if at.sidebar.radio:
        at.sidebar.radio[0].set_value("Upload Custom CSV")
        at.run(timeout=30)

    # Upload empty / malformed CSV
    bad_csv_bytes = b"bad,columns,only\n1,2,3\n"
    if at.sidebar.file_uploader:
        at.sidebar.file_uploader[0].upload("corrupt.csv", bad_csv_bytes)
        at.run(timeout=30)

    # App must NOT crash with unhandled exception
    assert not at.exception
    # Should display st.error or st.warning
    assert len(at.error) > 0 or len(at.warning) > 0
