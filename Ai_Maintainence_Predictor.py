from pathlib import Path
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Predictive Maintenance Platform",
    layout="wide",
    page_icon="⚙"
)

# --------------------------------------------------
# CUSTOM CSS (Enterprise Styling)
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.metric-box {
    background: linear-gradient(145deg, #1f2630, #11161d);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}
.big-font {
    font-size: 30px !important;
    font-weight: 700;
}
.small-font {
    font-size: 14px;
    color: #9aa4b2;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    rf = joblib.load(BASE_DIR / "rul_model.pkl")
    iso = joblib.load(BASE_DIR / "anomaly_model.pkl")
    feature_cols = joblib.load(BASE_DIR / "feature_cols.pkl")
    return rf, iso, feature_cols

rf, iso, feature_cols = load_models()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<h1 style='text-align:center;'>⚙ AI-Driven Predictive Maintenance Control Center</h1>
<p style='text-align:center;color:#9aa4b2;'>Real-Time Degradation Monitoring • Failure Forecasting • Anomaly Intelligence</p>
""", unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙ System Controls")

uploaded_file = st.sidebar.file_uploader("Upload Engine Dataset")

threshold = st.sidebar.slider(
    "Anomaly Sensitivity Threshold",
    -0.5, 0.5, 0.0, 0.01
)

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    X = df[feature_cols]

    df["Predicted_RUL"] = rf.predict(X)
    df["Anomaly_Score"] = iso.decision_function(X)
    df["Anomaly_Flag"] = np.where(df["Anomaly_Score"] < threshold, -1, 1)

    engine_ids = df["id"].unique()
    selected_engine = st.sidebar.selectbox("Select Engine", engine_ids)

    engine_df = df[df["id"] == selected_engine].sort_values("cycle")

    latest_rul = max(0, engine_df["Predicted_RUL"].iloc[-1])
    latest_score = engine_df["Anomaly_Score"].iloc[-1]

    failure_cycle = engine_df["cycle"].max()
    anomaly_points = engine_df[engine_df["Anomaly_Flag"] == -1]

    if not anomaly_points.empty:
        first_anomaly = anomaly_points["cycle"].min()
        lead_time = max(0, failure_cycle - first_anomaly)
    else:
        lead_time = 0

    # --------------------------------------------------
    # KPI ROW
    # --------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div class="metric-box">
        <div class="small-font">Predicted Remaining Useful Life</div>
        <div class="big-font">{int(latest_rul)} cycles</div>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-box">
        <div class="small-font">Current Anomaly Score</div>
        <div class="big-font">{latest_score:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="metric-box">
        <div class="small-font">Early Warning Lead Time</div>
        <div class="big-font">{lead_time} cycles</div>
    </div>
    """, unsafe_allow_html=True)

    risk_level = "HIGH RISK" if latest_score < threshold else "NORMAL"
    col4.markdown(f"""
    <div class="metric-box">
        <div class="small-font">System Status</div>
        <div class="big-font">{risk_level}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --------------------------------------------------
    # GAUGE METER
    # --------------------------------------------------
    st.subheader("Engine Risk Indicator")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_score,
        title={'text': "Anomaly Risk Meter"},
        gauge={
            'axis': {'range': [-0.5, 0.5]},
            'bar': {'color': "red" if latest_score < threshold else "green"},
            'steps': [
                {'range': [-0.5, threshold], 'color': "#3d0000"},
                {'range': [threshold, 0.5], 'color': "#002e00"}
            ],
        }
    ))

    st.plotly_chart(fig_gauge, width="stretch")

    st.divider()

    # --------------------------------------------------
    # TREND SECTION
    # --------------------------------------------------
    colA, colB = st.columns(2)

    with colA:
        st.subheader("RUL Degradation Trend")
        fig_rul = px.line(
            engine_df,
            x="cycle",
            y="Predicted_RUL",
            template="plotly_dark"
        )
        fig_rul.update_layout(height=400)
        st.plotly_chart(fig_rul, width="stretch")

    with colB:
        st.subheader("Anomaly Progression")
        fig_anomaly = px.line(
            engine_df,
            x="cycle",
            y="Anomaly_Score",
            template="plotly_dark"
        )
        fig_anomaly.add_hline(y=threshold, line_dash="dash")
        fig_anomaly.update_layout(height=400)
        st.plotly_chart(fig_anomaly, width="stretch")

    st.divider()

    # --------------------------------------------------
    # ALERT TABLE
    # --------------------------------------------------
    st.subheader("Detected Anomaly Events")

    if not anomaly_points.empty:
        st.dataframe(
            anomaly_points[["cycle", "Predicted_RUL", "Anomaly_Score"]],
            width="stretch"
        )
    else:
        st.success("No anomalies detected for selected engine.")

else:
    st.info("Upload engine dataset to initialize monitoring system.")