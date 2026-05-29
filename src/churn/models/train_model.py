"""Step 3 of the pipeline: train models and track them with MLflow.

Each model is trained as its own MLflow run under the same experiment, so
the local MLflow UI shows several runs side by side. Parameters, metrics,
and the fitted model are all logged.

Run all three baselines::

    python -m churn.models.train_model

Train a single model (used by the MLproject entry point)::

    python -m churn.models.train_model --model xgboost
"""

import argparse

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn import config


def load_split():
    """Load the processed train/test matrices."""
    x_train = pd.read_parquet(config.X_TRAIN)
    x_test = pd.read_parquet(config.X_TEST)
    y_train = pd.read_parquet(config.Y_TRAIN)[config.TARGET]
    y_test = pd.read_parquet(config.Y_TEST)[config.TARGET]
    return x_train, x_test, y_train, y_test


def build_estimator(name: str):
    """Return an (estimator, params) pair for the requested model name."""
    if name == "logreg":
        params = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}
        # Logistic regression needs scaling; trees do not.
        est = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**params)),
            ]
        )
        return est, params
    if name == "random_forest":
        params = {
            "n_estimators": 300,
            "max_depth": 12,
            "class_weight": "balanced",
            "random_state": config.RANDOM_STATE,
        }
        return RandomForestClassifier(**params), params
    if name == "xgboost":
        params = {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "logloss",
            "random_state": config.RANDOM_STATE,
        }
        return XGBClassifier(**params), params
    raise ValueError(f"Unknown model: {name}")


def evaluate(model, x_test, y_test) -> dict:
    """Compute the metrics we care about for an imbalanced churn problem."""
    preds = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def log_model_artifact(name: str, model) -> None:
    """Log the fitted model with the right MLflow flavour."""
    if name == "xgboost":
        mlflow.xgboost.log_model(model, name="model")
    else:
        mlflow.sklearn.log_model(model, name="model")


def train_one(name: str, x_train, x_test, y_train, y_test) -> dict:
    """Train a single model inside its own MLflow run."""
    est, params = build_estimator(name)
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", name)
        mlflow.log_params(params)
        est.fit(x_train, y_train)
        metrics = evaluate(est, x_test, y_test)
        mlflow.log_metrics(metrics)
        log_model_artifact(name, est)
        print(f"[{name}] " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return metrics


def main() -> None:
    """Train the requested model(s) under the churn experiment."""
    parser = argparse.ArgumentParser(description="Train churn models.")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "logreg", "random_forest", "xgboost"],
        help="Which model to train (default: all three).",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    x_train, x_test, y_train, y_test = load_split()
    names = (
        ["logreg", "random_forest", "xgboost"] if args.model == "all" else [args.model]
    )
    for name in names:
        train_one(name, x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
