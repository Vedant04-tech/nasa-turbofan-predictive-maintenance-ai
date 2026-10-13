"""
NASA Turbofan Predictive Maintenance AI Package.

Provides robust data validation, feature engineering, dual-model inference,
and prognostic evaluation for industrial rotating machinery.
"""

from turbofan_ai.data import (
    ALL_RAW_COLS,
    LOW_VARIANCE_SENSORS,
    RAW_METADATA_COLS,
    ROLLING_SENSORS,
    compute_rolling_features,
    validate_and_prepare_data,
)
from turbofan_ai.evaluation import (
    calculate_early_warning_lead_time,
    compute_regression_metrics,
    get_anomaly_summary,
)
from turbofan_ai.models import (
    load_model_artifacts,
    predict_anomaly,
    predict_rul,
)

__version__ = "1.0.0"
__all__ = [
    "ALL_RAW_COLS",
    "LOW_VARIANCE_SENSORS",
    "RAW_METADATA_COLS",
    "ROLLING_SENSORS",
    "compute_rolling_features",
    "validate_and_prepare_data",
    "load_model_artifacts",
    "predict_anomaly",
    "predict_rul",
    "calculate_early_warning_lead_time",
    "compute_regression_metrics",
    "get_anomaly_summary",
]
