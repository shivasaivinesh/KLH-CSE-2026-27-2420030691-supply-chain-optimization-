"""
Central configuration for the ML-Based Predictive Analytics for Supply Chain
Optimization project.

Every module reads paths and constants from here so that the whole pipeline can
be re-run with a single command and re-pointed at another machine by setting
the SC_DATA_ROOT environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw datasets are large (>200 MB in total) and are deliberately NOT committed
# to this repository.  Set SC_DATA_ROOT to the folder that holds them.
DATA_ROOT = Path(os.environ.get("SC_DATA_ROOT", PROJECT_ROOT / "data"))
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"

for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset file names (as produced by src/fetch_data.py)
# --------------------------------------------------------------------------
DATACO_CSV = RAW_DIR / "DataCoSupplyChainDataset.csv"
STORE_DEMAND_CSV = RAW_DIR / "store_item_demand_train.csv"
ONLINE_RETAIL_CSV = RAW_DIR / "online_retail_II.csv"

# --------------------------------------------------------------------------
# Experimental protocol
# --------------------------------------------------------------------------
RANDOM_STATE = 42

# Demand forecasting (Dataset D2 - Store Item Demand)
FORECAST_HORIZONS = (7, 30, 90)          # days, as stated in the Review-I plan
TEST_DAYS = 180                          # held-out tail used for the 90-day test
VALID_DAYS = 90                          # tail of the training window for tuning
N_ARIMA_SERIES = 12                      # representative series for the ARIMA baseline

# Inventory optimisation
SERVICE_LEVEL = 0.95                     # -> Z = 1.645
Z_SCORE = 1.645
ORDERING_COST = 50.0                     # ₹ / order  (S)
UNIT_COST = 12.0                         # ₹ / unit   (C)
ANNUAL_HOLDING_RATE = 0.20               # h  -> H = h * C
LEAD_TIME_DAYS = 7                       # replenishment lead time L

# Delay-risk classifier (Dataset D1 - DataCo)
RISK_TEST_FRACTION = 0.20

# Reproducibility / determinism
N_JOBS = int(os.environ.get("SC_N_JOBS", "-1"))
