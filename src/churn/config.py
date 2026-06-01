"""Central configuration: paths, column groups, and constants.

Keeping these in one place means the data, feature, model, and explain
scripts all agree on where things live and what the columns mean.
"""

from pathlib import Path

# Project layout ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATASET = RAW_DIR / "ECommerceDataset.xlsx"
RAW_SHEET = "E Comm"

# Cleaned full table and the train/test feature matrices.
CLEAN_DATA = INTERIM_DIR / "churn_clean.parquet"
X_TRAIN = PROCESSED_DIR / "X_train.parquet"
X_TEST = PROCESSED_DIR / "X_test.parquet"
Y_TRAIN = PROCESSED_DIR / "y_train.parquet"
Y_TEST = PROCESSED_DIR / "y_test.parquet"

# Columns -----------------------------------------------------------------
TARGET = "Churn"
ID_COL = "CustomerID"

# Numeric columns that contain missing values in the raw file.
NUMERIC_COLS = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
    "CityTier",
    "Complain",
]

CATEGORICAL_COLS = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

# Reproducibility ---------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# MLflow ------------------------------------------------------------------
EXPERIMENT_NAME = "retailgenius-churn"
REGISTERED_MODEL_NAME = "churn-model"
TRACKING_URI = f"file:///{(PROJECT_ROOT / 'mlruns').as_posix()}"
