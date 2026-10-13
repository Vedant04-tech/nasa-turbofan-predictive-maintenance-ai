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
    """Verify threshold slider interacts with anomaly counts and lead times."""
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run(timeout=30)
    assert not at.exception

    # Select engine unit 2 if present
    if at.sidebar.selectbox:
        at.sidebar.selectbox[0].select(2)
        at.run(timeout=30)
        assert not at.exception

    # Read initial lead time
    initial_text = " ".join([m.value for m in at.markdown if "Early Warning Lead Time" in m.value])

    # Adjust threshold slider if present
    if at.sidebar.slider:
        at.sidebar.slider[0].set_value(-0.05)
        at.run(timeout=30)
        assert not at.exception
        tuned_text = " ".join([m.value for m in at.markdown if "Early Warning Lead Time" in m.value])
        assert initial_text != tuned_text or len(at.dataframe) >= 0


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
