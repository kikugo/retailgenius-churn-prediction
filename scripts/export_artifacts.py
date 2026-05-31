"""Export MLflow outputs and the trained model into the repository.

The assignment asks the repo to contain the generated models and MLflow
outputs. The ``mlruns/`` tracking store is gitignored (it is large and
machine-specific), so this script copies the committable bits out:

* ``reports/mlflow_runs.csv`` -- one row per run with params and metrics.
* ``models/churn-model/`` -- the Production model artifacts.

Run after training + registering::

    python -m scripts.export_artifacts
"""

import shutil

import mlflow

from churn import config


def export_runs() -> None:
    """Dump each run's params and metrics to a CSV.

    We keep only the model name, parameters, and metrics. Machine-specific
    columns (artifact paths, absolute URIs, start/end timestamps, host user)
    are dropped so the export is portable and reproducible.
    """
    mlflow.set_tracking_uri(config.TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[config.EXPERIMENT_NAME])

    keep = [c for c in runs.columns if c.startswith(("params.", "metrics."))]
    name_col = "tags.mlflow.runName"
    cols = ([name_col] if name_col in runs.columns else []) + keep
    tidy = runs[cols].rename(columns={name_col: "model"})

    out = config.REPORTS_DIR / "mlflow_runs.csv"
    tidy.to_csv(out, index=False)
    print(f"Wrote {out} ({len(tidy)} runs, {len(tidy.columns)} columns)")


def export_model() -> None:
    """Copy the Production model artifacts into models/."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    # Resolve the Production version, then download its artifacts.
    prod = [
        mv
        for mv in client.search_model_versions(
            f"name = '{config.REGISTERED_MODEL_NAME}'"
        )
        if mv.current_stage == "Production"
    ]
    if not prod:
        print("No Production model to export.")
        return
    dest = config.MODELS_DIR / config.REGISTERED_MODEL_NAME
    if dest.exists():
        shutil.rmtree(dest)
    local = mlflow.artifacts.download_artifacts(prod[0].source)
    shutil.copytree(local, dest)
    print(f"Copied Production model -> {dest}")


def main() -> None:
    """Export both the run table and the model artifacts."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    export_runs()
    export_model()


if __name__ == "__main__":
    main()
