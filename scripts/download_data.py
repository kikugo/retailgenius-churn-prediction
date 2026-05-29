"""Download the E-Commerce churn dataset into ``data/raw``.

The raw data is kept out of Git, so this script makes the project
reproducible: anyone who clones the repo runs it once to fetch the file.

The dataset is the public Kaggle "E Commerce Dataset" (5,630 rows). We pull
it from a stable GitHub mirror to avoid the Kaggle API/login dance.
"""

import urllib.request

from churn.config import RAW_DATASET, RAW_DIR

MIRROR_URL = (
    "https://raw.githubusercontent.com/Leangonplu/"
    "Ecommerce_Customer_Churn_Analysis_and_Prediction/main/"
    "E%20Commerce%20Dataset.xlsx"
)


def download() -> None:
    """Fetch the dataset to ``RAW_DATASET`` if it is not already there."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATASET.exists():
        print(f"Dataset already present: {RAW_DATASET}")
        return
    print(f"Downloading dataset from {MIRROR_URL}")
    urllib.request.urlretrieve(MIRROR_URL, RAW_DATASET)
    print(f"Saved to {RAW_DATASET}")


if __name__ == "__main__":
    download()
