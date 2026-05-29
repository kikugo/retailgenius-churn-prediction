# RetailGenius Churn Prediction

Predicting which e-commerce customers are about to leave, built for the EPITA
AI Project Methodology project (Parts 2 and 3).

We were graded on how the project is put together rather than how accurate it
is, so most of the effort went into the things that make a project easy to pick
up and rerun: a reproducible environment, a clean layout, PEP8, docs that
generate themselves, MLflow for tracking and serving, and SHAP to explain what
the model is doing.

## Stack

- Python 3.11, dependencies managed with [uv](https://docs.astral.sh/uv/)
- scikit-learn + XGBoost for modelling
- MLflow for tracking, the model registry, and local serving
- SHAP for explainability (Part 3)
- black + flake8 (via pre-commit) for PEP8, Sphinx for docs

## Project layout

```
src/churn/
  data/make_dataset.py      # load + clean the raw Excel file
  features/build_features.py # encode + train/test split
  models/train_model.py      # train + MLflow tracking (3 runs)
  models/register_model.py   # promote best run to the registry
  models/predict_model.py    # load Production model + score
  explain/shap_explain.py    # SHAP plots (Part 3)
  serving/sample_request.py  # call the local inference server
```

## Setup

```bash
uv sync                      # create the env from pyproject.toml + uv.lock
uv run python scripts/download_data.py   # fetch dataset into data/raw
```

## Run the pipeline

```bash
uv run python -m churn.data.make_dataset
uv run python -m churn.features.build_features
uv run python -m churn.models.train_model        # 3 MLflow runs
uv run python -m churn.models.register_model      # best -> Production
uv run python -m churn.models.predict_model       # score the test set
```

Inspect the runs:

```bash
uv run mlflow ui   # then open http://127.0.0.1:5000
```

## Serve the model locally

```bash
uv run mlflow models serve -m "models:/churn-model/Production" -p 5001 --env-manager local
# in another shell:
uv run python -m churn.serving.sample_request
```

## Explainability (Part 3)

```bash
uv run python -m churn.explain.shap_explain   # writes plots to reports/figures
```

## Checks

```bash
uv run pre-commit run --all-files   # black + flake8
uv run pytest                       # smoke tests
cd docs && uv run make html         # build the HTML docs
```

## Run as an MLflow Project

```bash
uv run mlflow run . -P model=xgboost --env-manager local
```

## Dataset

The public [E Commerce Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
from Kaggle: 5,630 rows, 20 columns, a binary `Churn` target. To skip the
Kaggle login, `scripts/download_data.py` grabs a copy from a GitHub mirror. The
raw file is kept out of Git on purpose.
