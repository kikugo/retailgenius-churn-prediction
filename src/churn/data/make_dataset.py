"""Step 1 of the pipeline: load the raw Excel file and clean it.

Reads ``data/raw/ECommerceDataset.xlsx``, fixes the known data-quality
issues (missing values and inconsistent category labels), and writes a
tidy table to ``data/interim/churn_clean.parquet``.

Run with::

    python -m churn.data.make_dataset
"""

import pandas as pd

from churn import config

# A handful of categories are spelled inconsistently in the raw file
# (e.g. "Mobile Phone" vs "Phone"). We collapse them here.
CATEGORY_FIXES = {
    "PreferredLoginDevice": {"Mobile Phone": "Phone"},
    "PreferredPaymentMode": {
        "COD": "Cash on Delivery",
        "CC": "Credit Card",
    },
    "PreferedOrderCat": {"Mobile": "Mobile Phone"},
}


def load_raw() -> pd.DataFrame:
    """Read the raw dataset from the Excel sheet."""
    if not config.RAW_DATASET.exists():
        raise FileNotFoundError(
            f"{config.RAW_DATASET} not found. "
            "Run `python scripts/download_data.py` first."
        )
    return pd.read_excel(config.RAW_DATASET, sheet_name=config.RAW_SHEET)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numbers, normalise categories, drop the ID column."""
    df = df.copy()

    # Missing numeric values: fill with the column median. The raw file has
    # gaps in Tenure, OrderCount, DaySinceLastOrder, etc.
    for col in config.NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Normalise inconsistent category spellings.
    for col, mapping in CATEGORY_FIXES.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    # CustomerID carries no signal and would leak as a feature.
    if config.ID_COL in df.columns:
        df = df.drop(columns=config.ID_COL)

    return df


def main() -> None:
    """Load, clean, and persist the interim table."""
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    print(f"Loaded raw data: {df.shape}")
    df = clean(df)
    print(f"Cleaned data: {df.shape}, missing values left: {df.isnull().sum().sum()}")
    df.to_parquet(config.CLEAN_DATA, index=False)
    print(f"Wrote {config.CLEAN_DATA}")


if __name__ == "__main__":
    main()
