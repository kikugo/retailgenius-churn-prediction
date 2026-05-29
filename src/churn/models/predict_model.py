"""Step 4 of the pipeline: load the registered model and run inference.

Pulls the ``Production`` version of ``churn-model`` from the registry and
scores the test set (or any parquet of features with the same schema).

Run with::

    python -m churn.models.predict_model
"""

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from churn import config


def load_production_model():
    """Load the Production-stage model from the registry."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    uri = f"models:/{config.REGISTERED_MODEL_NAME}/Production"
    return mlflow.pyfunc.load_model(uri)


def predict(features: pd.DataFrame) -> pd.DataFrame:
    """Return churn predictions for a feature matrix."""
    model = load_production_model()
    preds = model.predict(features)
    out = features.copy()
    out["churn_prediction"] = preds
    return out


def main() -> None:
    """Score a parquet file of features (defaults to the test set)."""
    parser = argparse.ArgumentParser(description="Score churn predictions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=config.X_TEST,
        help="Parquet of features to score (default: processed test set).",
    )
    args = parser.parse_args()

    features = pd.read_parquet(args.input)
    result = predict(features)
    n_churn = int(result["churn_prediction"].sum())
    print(f"Scored {len(result)} rows; predicted churn for {n_churn}.")
    print(result["churn_prediction"].value_counts())


if __name__ == "__main__":
    main()
