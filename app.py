"""
NASA Turbofan Predictive Maintenance Control Center.
Streamlit web application for real-time degradation monitoring, failure forecasting,
and anomaly intelligence powered by turbofan_ai package.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src is in python path if running directly
SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from turbofan_ai.data import validate_and_prepare_data
from turbofan_ai.evaluation import get_anomaly_summary
from turbofan_ai.models import (
    load_model_artifacts,
    predict_anomaly,
    predict_rul,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="NASA Turbofan Predictive Maintenance AI",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM CSS STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e2633 0%, #111722 100%);
        border: 1px solid #2a374a;
        padding: 22px 16px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3f5573;
    }
    .metric-title {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b9bb4;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #f1f5f9;
    }
    .status-normal {
        color: #22c55e !important;
    }
    .status-warning {
        color: #f59e0b !important;
    }
    .status-critical {
        color: #ef4444 !important;
    }
    .subtext {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# MODEL LOADING (CACHED)
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading prognostic models...")
def get_models():
    """Loads and caches trained model artifacts."""
    return load_model_artifacts()


def find_sample_dataset() -> Optional[Path]:
    """Finds canonical sample dataset path across standard locations."""
    candidates = [
        Path(__file__).resolve().parent / "data" / "samples" / "engine_samples.csv",
        Path.cwd() / "data" / "samples" / "engine_samples.csv",
        Path.cwd() / "engine_samples.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# --------------------------------------------------
# MAIN APPLICATION CONTROLLER
# --------------------------------------------------
def main():
    # 1. Header Banner
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="margin-bottom: 4px; font-weight: 800;">⚙ NASA Turbofan Predictive Maintenance Control Center</h1>
            <p style="color: #94a3b8; font-size: 16px;">
                Physics-Informed Fleet Degradation Forecasting • Early Anomaly Detection • C-MAPSS Telemetry Analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Load Models
    try:
        rf_model, iso_model, feature_cols = get_models()
    except Exception as e:
        st.error(
            f"❌ **Model Artifact Initialization Failure**: {e}\n\n"
            "Please check that model artifacts (`rul_model.pkl`, `anomaly_model.pkl`, `feature_cols.pkl`) "
            "exist in the `models/` directory. If using Git LFS, ensure `git lfs pull` was executed."
        )
        st.stop()

    # 3. Sidebar Controls
    st.sidebar.header("⚙ System Controls")

    # Dataset acquisition options
    sample_file_path = find_sample_dataset()
    has_sample = sample_file_path is not None

    data_source = st.sidebar.radio(
        "Telemetry Data Source",
        options=["Upload Custom CSV", "Load Benchmark Sample Dataset"] if has_sample else ["Upload Custom CSV"],
        index=1 if has_sample else 0,
    )

    uploaded_file = None
    raw_df: Optional[pd.DataFrame] = None

    if data_source == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Engine Telemetry (.csv)",
            type=["csv"],
            help="Upload a CSV containing NASA C-MAPSS sensor channels or processed engine cycles.",
        )
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.sidebar.error(f"Failed to read CSV: {e}")
    else:
        if sample_file_path and sample_file_path.exists():
            try:
                raw_df = pd.read_csv(sample_file_path)
                st.sidebar.success(f"✓ Benchmark dataset loaded ({len(raw_df)} cycles)")
            except Exception as e:
                st.sidebar.error(f"Failed to load sample dataset: {e}")

    threshold = st.sidebar.slider(
        "Anomaly Sensitivity Threshold",
        min_value=-0.50,
        max_value=0.50,
        value=0.00,
        step=0.01,
        help="Isolation Forest decision score boundary. Scores below this threshold trigger an anomaly flag.",
    )

    with st.sidebar.expander("ℹ️ Model Architecture & Specs"):
        st.markdown(
            f"""
            - **RUL Regressor**: Random Forest (100 Estimators)
            - **Anomaly Detector**: Isolation Forest (Contamination=0.05)
            - **Total Input Features**: {len(feature_cols)}
            - **Causal Rolling Window**: 5 cycles
            - **Target Capping**: RUL capped at 125 cycles
            """
        )

    # 4. Processing & Dashboard
    if raw_df is None:
        st.info(
            "👋 Welcome to the Turbofan Health Monitoring System.\n\n"
            "Please select **'Load Benchmark Sample Dataset'** from the sidebar or upload your own engine CSV telemetry to begin."
        )
        return

    # Data Validation and Preparation
    try:
        prepared_df, warnings = validate_and_prepare_data(raw_df, feature_cols=feature_cols)
    except ValueError as ve:
        st.error(f"⚠️ **Data Validation Error**: {ve}")
        return
    except Exception as exc:
        st.error(f"❌ **Unexpected Error during Data Preparation**: {exc}")
        return

    if warnings:
        with st.expander("⚠️ Data Preparation Warnings", expanded=False):
            for w in warnings:
                st.caption(f"• {w}")

    # Inference Execution
    X = prepared_df[feature_cols]
    prepared_df["Predicted_RUL"] = predict_rul(rf_model, X)
    scores, flags = predict_anomaly(iso_model, X, threshold=threshold)
    prepared_df["Anomaly_Score"] = scores
    prepared_df["Anomaly_Flag"] = flags

    # Engine Selection
    engine_ids = sorted(prepared_df["id"].unique())
    selected_engine = st.sidebar.selectbox("Select Engine Unit ID", engine_ids)

    engine_df = prepared_df[prepared_df["id"] == selected_engine].sort_values("cycle").reset_index(drop=True)
    summary = get_anomaly_summary(engine_df, threshold=threshold, score_col="Anomaly_Score")

    latest_cycle = int(engine_df["cycle"].iloc[-1])
    latest_rul = float(engine_df["Predicted_RUL"].iloc[-1])
    latest_score = float(summary["latest_score"])
    lead_time = int(summary["lead_time"])
    status = summary["status"]

    # 5. KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Remaining Useful Life</div>
                <div class="metric-value">{int(latest_rul)} <span style="font-size:16px;color:#94a3b8;">cycles</span></div>
                <div class="subtext">Current Cycle: #{latest_cycle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        score_color_class = "status-critical" if latest_score < threshold else "status-normal"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Anomaly Decision Score</div>
                <div class="metric-value {score_color_class}">{latest_score:+.3f}</div>
                <div class="subtext">Cutoff Boundary: {threshold:+.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Early Warning Lead Time</div>
                <div class="metric-value">{lead_time} <span style="font-size:16px;color:#94a3b8;">cycles</span></div>
                <div class="subtext">{"First Trigger: Cycle #" + str(summary['first_anomaly_cycle']) if summary['first_anomaly_cycle'] else "No Anomaly Triggered"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        status_class = "status-critical" if status == "HIGH RISK" else "status-normal"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Health Status</div>
                <div class="metric-value {status_class}">{status}</div>
                <div class="subtext">{summary['anomaly_count']} anomaly cycles detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 6. Interactive Visualizations
    tab_overview, tab_sensors, tab_anomalies = st.tabs(
        ["📊 Prognostics & Risk Overview", "📈 Sensor Telemetry Explorer", "📋 Anomaly Event Log"]
    )

    with tab_overview:
        col_gauge, col_trend = st.columns([1, 2])

        with col_gauge:
            st.subheader("Subsystem Risk Meter")
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=latest_score,
                    number={"valueformat": "+.3f"},
                    title={"text": f"Engine {selected_engine} Risk Score", "font": {"size": 16}},
                    gauge={
                        "axis": {"range": [-0.5, 0.5]},
                        "bar": {"color": "#ef4444" if latest_score < threshold else "#22c55e"},
                        "steps": [
                            {"range": [-0.5, threshold], "color": "rgba(239, 68, 68, 0.25)"},
                            {"range": [threshold, 0.5], "color": "rgba(34, 197, 94, 0.25)"},
                        ],
                        "threshold": {
                            "line": {"color": "#f59e0b", "width": 3},
                            "thickness": 0.8,
                            "value": threshold,
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                margin=dict(l=30, r=30, t=50, b=20),
            )
            st.plotly_chart(fig_gauge, width="stretch")

        with col_trend:
            st.subheader("RUL Degradation Trajectory")
            fig_rul = go.Figure()
            fig_rul.add_trace(
                go.Scatter(
                    x=engine_df["cycle"],
                    y=engine_df["Predicted_RUL"],
                    mode="lines",
                    name="Predicted RUL",
                    line=dict(color="#38bdf8", width=2.5),
                )
            )
            fig_rul.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Operating Cycle",
                yaxis_title="Predicted RUL (cycles)",
                height=340,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_rul, width="stretch")

        # Anomaly Progression Line Chart
        st.subheader("Degradation Anomaly Progression")
        fig_anomaly = go.Figure()
        fig_anomaly.add_trace(
            go.Scatter(
                x=engine_df["cycle"],
                y=engine_df["Anomaly_Score"],
                mode="lines",
                name="Anomaly Score",
                line=dict(color="#94a3b8", width=1.5),
            )
        )
        # Highlight anomaly flags
        anomalies_only = engine_df[engine_df["Anomaly_Flag"] == -1]
        if not anomalies_only.empty:
            fig_anomaly.add_trace(
                go.Scatter(
                    x=anomalies_only["cycle"],
                    y=anomalies_only["Anomaly_Score"],
                    mode="markers",
                    name="Anomalies Flagged",
                    marker=dict(color="#ef4444", size=7, symbol="x"),
                )
            )

        fig_anomaly.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"Decision Cutoff ({threshold:+.2f})",
            annotation_position="bottom right",
        )
        fig_anomaly.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Operating Cycle",
            yaxis_title="Isolation Forest Decision Score",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_anomaly, width="stretch")

        st.caption(
            "📌 **Lead Time Semantics**: In run-to-failure benchmarks, lead time represents cycles from initial trigger "
            "to functional breakdown. For active in-service units, it indicates the elapsed monitoring duration since first detection."
        )

    with tab_sensors:
        st.subheader("Multivariate Sensor Telemetry Analysis")
        sensor_options = [c for c in engine_df.columns if c.startswith("sensor_") and not c.endswith("_rolling_std")]
        selected_sensor = st.selectbox(
            "Select Sensor Channel to Inspect",
            options=sensor_options,
            index=0 if sensor_options else None,
        )

        if selected_sensor:
            fig_sensor = px.line(
                engine_df,
                x="cycle",
                y=selected_sensor,
                title=f"Telemetry Signature: {selected_sensor} (Engine {selected_engine})",
                template="plotly_dark",
            )
            rolling_mean_col = f"{selected_sensor}_rolling_mean"
            if rolling_mean_col in engine_df.columns:
                fig_sensor.add_trace(
                    go.Scatter(
                        x=engine_df["cycle"],
                        y=engine_df[rolling_mean_col],
                        mode="lines",
                        name="Causal Rolling Mean (w=5)",
                        line=dict(color="#fbbf24", dash="dash"),
                    )
                )
            fig_sensor.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Cycle",
                yaxis_title="Sensor Reading",
            )
            st.plotly_chart(fig_sensor, width="stretch")

    with tab_anomalies:
        st.subheader("Detailed Anomaly Detection Records")
        anom_records = engine_df[engine_df["Anomaly_Flag"] == -1][
            ["cycle", "Predicted_RUL", "Anomaly_Score"]
        ].copy()

        if not anom_records.empty:
            anom_records["Predicted_RUL"] = anom_records["Predicted_RUL"].round(1)
            anom_records["Anomaly_Score"] = anom_records["Anomaly_Score"].round(4)
            st.dataframe(anom_records, width="stretch")
            csv_data = anom_records.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Anomaly Incident Log (CSV)",
                data=csv_data,
                file_name=f"engine_{selected_engine}_anomalies.csv",
                mime="text/csv",
            )
        else:
            st.success(f"✓ No degradation anomalies detected for Engine Unit #{selected_engine} at threshold {threshold}.")


if __name__ == "__main__":
    main()
