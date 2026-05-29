"""Pick the best run and promote it to the MLflow Model Registry.

Looks across all runs in the experiment, selects the one with the highest
ROC-AUC, registers its logged model under ``churn-model`` and moves that
version into the ``Production`` stage.

Run with::

    python -m churn.models.register_model
"""

import mlflow
from mlflow.tracking import MlflowClient

from churn import config


def best_run(client: MlflowClient):
    """Return the run with the highest test ROC-AUC."""
    experiment = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{config.EXPERIMENT_NAME}' not found. Train first."
        )
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found. Run train_model first.")
    return runs[0]


def main() -> None:
    """Register the best model and move it to Production."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = MlflowClient()

    run = best_run(client)
    auc = run.data.metrics["roc_auc"]
    model_type = run.data.params.get("model_type", "unknown")
    print(f"Best run: {run.info.run_id} ({model_type}, roc_auc={auc:.3f})")

    model_uri = f"runs:/{run.info.run_id}/model"
    version = mlflow.register_model(model_uri, config.REGISTERED_MODEL_NAME)
    print(f"Registered {config.REGISTERED_MODEL_NAME} v{version.version}")

    client.transition_model_version_stage(
        name=config.REGISTERED_MODEL_NAME,
        version=version.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Moved v{version.version} to Production")


if __name__ == "__main__":
    main()
