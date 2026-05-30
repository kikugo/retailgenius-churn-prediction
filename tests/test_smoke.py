"""Smoke tests for the churn pipeline.

These do not check model quality (that is explicitly not the goal). They
just confirm the moving parts wire together: cleaning fills nulls, feature
building produces an aligned split, and a model can fit and predict.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from churn.data.make_dataset import clean
from churn.features.build_features import add_derived_features, encode


def _raw_frame() -> pd.DataFrame:
    """A tiny raw-shaped frame with a missing value and an ID column."""
    return pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4],
            "Churn": [0, 1, 0, 1],
            "Tenure": [1.0, None, 5.0, 8.0],
            "OrderCount": [2.0, 3.0, 1.0, 4.0],
            "CouponUsed": [1.0, 0.0, 2.0, 1.0],
            "CashbackAmount": [120.0, 80.0, 200.0, 50.0],
            "Gender": ["Male", "Female", "Male", "Female"],
        }
    )


def test_clean_fills_nulls_and_drops_id():
    cleaned = clean(_raw_frame())
    assert "CustomerID" not in cleaned.columns
    assert cleaned["Tenure"].isnull().sum() == 0


def test_features_encode_and_align():
    df = add_derived_features(clean(_raw_frame()))
    encoded = encode(df)
    assert "CouponPerOrder" in encoded.columns
    # Gender should be one-hot encoded away from an object column.
    assert encoded.select_dtypes("object").empty


def test_model_fits_and_predicts():
    df = encode(add_derived_features(clean(_raw_frame())))
    y = df["Churn"]
    x = df.drop(columns="Churn")
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(x, y)
    preds = model.predict(x)
    assert len(preds) == len(y)
