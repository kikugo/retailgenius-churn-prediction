"""Step 2 of the pipeline: turn the clean table into model-ready matrices.

One-hot encodes the categorical columns, adds a couple of derived
features, and writes a stratified train/test split to ``data/processed``.

Run with::

    python -m churn.features.build_features
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few ratio features that often help on this dataset."""
    df = df.copy()
    # Coupons used per order: heavy coupon reliance can signal price
    # sensitivity, which tends to correlate with churn.
    df["CouponPerOrder"] = df["CouponUsed"] / (df["OrderCount"] + 1)
    # Cashback per order, a rough proxy for how rewarded a customer feels.
    df["CashbackPerOrder"] = df["CashbackAmount"] / (df["OrderCount"] + 1)
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the categorical columns."""
    present = [c for c in config.CATEGORICAL_COLS if c in df.columns]
    return pd.get_dummies(df, columns=present, drop_first=True)


def main() -> None:
    """Build features and write the stratified train/test split."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(config.CLEAN_DATA)

    df = add_derived_features(df)
    df = encode(df)

    y = df[config.TARGET]
    x = df.drop(columns=config.TARGET)
    # Booleans from get_dummies -> ints, so the serialized schema is clean.
    x = x.astype({c: "int" for c in x.select_dtypes("bool").columns})

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    x_train.to_parquet(config.X_TRAIN, index=False)
    x_test.to_parquet(config.X_TEST, index=False)
    y_train.to_frame().to_parquet(config.Y_TRAIN, index=False)
    y_test.to_frame().to_parquet(config.Y_TEST, index=False)

    print(f"Features: {x.shape[1]} columns")
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    print(f"Churn rate train={y_train.mean():.3f} test={y_test.mean():.3f}")


if __name__ == "__main__":
    main()
