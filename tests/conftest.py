"""
Pytest configuration, shared fixtures, and testing utilities.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from turbofan_ai.data import ROLLING_SENSORS


@pytest.fixture
def project_root() -> Path:
    return BASE_DIR


@pytest.fixture
def models_dir(project_root) -> Path:
    return project_root / "models"


@pytest.fixture
def sample_csv_path(project_root) -> Path:
    return project_root / "data" / "samples" / "engine_samples.csv"


@pytest.fixture
def sample_dataframe(sample_csv_path) -> pd.DataFrame:
    assert sample_csv_path.exists(), f"Sample CSV missing at {sample_csv_path}"
    return pd.read_csv(sample_csv_path)


@pytest.fixture
def synthetic_raw_engine_data() -> pd.DataFrame:
    """Creates synthetic multi-cycle raw telemetry matching NASA C-MAPSS schema without rolling features."""
    np.random.seed(42)
    n_cycles = 20
    rows = []
    for eng_id in [1, 2]:
        for c in range(1, n_cycles + 1):
            row = {
                "id": eng_id,
                "cycle": c,
                "op_set_1": np.random.normal(0, 0.01),
                "op_set_2": np.random.normal(0, 0.001),
                "op_set_3": 100.0,
            }
            for s in range(1, 22):
                # Add degrading trend on rolling sensors
                base = 500.0 + (c * 2.0 if f"sensor_{s}" in ROLLING_SENSORS else 0.0)
                row[f"sensor_{s}"] = base + np.random.normal(0, 0.5)
            rows.append(row)
    return pd.DataFrame(rows)
