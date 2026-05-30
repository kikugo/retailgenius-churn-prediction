"""Send a sample request to a locally served churn model.

First serve the registered model in another terminal::

    mlflow models serve -m "models:/churn-model/Production" -p 5001 --env-manager local

Then run this script to score a few rows from the test set against it::

    python -m churn.serving.sample_request
"""

import argparse
import json
import urllib.request

import pandas as pd

from churn import config

DEFAULT_ENDPOINT = "http://127.0.0.1:5001/invocations"


def build_payload(n_rows: int) -> dict:
    """Build an MLflow `dataframe_split` payload from the test set."""
    sample = pd.read_parquet(config.X_TEST).head(n_rows)
    return {
        "dataframe_split": {
            "columns": sample.columns.tolist(),
            "data": sample.values.tolist(),
        }
    }


def call_server(endpoint: str, payload: dict) -> str:
    """POST the payload to the MLflow serving endpoint and return the body."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")


def main() -> None:
    """Score a few rows against the running inference server."""
    parser = argparse.ArgumentParser(description="Call the churn server.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    payload = build_payload(args.rows)
    print(f"POST {args.rows} rows -> {args.endpoint}")
    print("Response:", call_server(args.endpoint, payload))


if __name__ == "__main__":
    main()
