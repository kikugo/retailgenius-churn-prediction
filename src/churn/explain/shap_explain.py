"""Part 3: explain the churn model with SHAP.

Loads the best tree-based model, builds a ``TreeExplainer``, and writes the
full set of plots required by the assignment to ``reports/figures``:

* single-point explanation (waterfall + force)
* all-points explanation (beeswarm + an interactive force plot)
* a summary plot per class
* mean-SHAP bar plot and a dependence plot

We explain a tree model (random forest by default) because ``TreeExplainer``
only works on tree ensembles, and its per-class output maps directly onto
the "summary plot for each class" requirement.

Run with::

    python -m churn.explain.shap_explain
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: render straight to files
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from churn import config  # noqa: E402

CHURN_CLASS = 1  # index of the "churned" class


def load_tree_model(model_type: str):
    """Load the best run of the requested tree model from MLflow."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("No experiment found. Train models first.")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"params.model_type = '{model_type}'",
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No '{model_type}' run found. Train it first.")
    run_id = runs[0].info.run_id
    uri = f"runs:/{run_id}/model"
    loader = mlflow.xgboost if model_type == "xgboost" else mlflow.sklearn
    print(f"Explaining {model_type} from run {run_id}")
    return loader.load_model(uri)


def to_class_list(shap_values) -> list:
    """Normalise SHAP output into a list of per-class 2D arrays.

    SHAP returns different shapes across versions/models (a list, a 3D
    array, or a single 2D array). This collapses all of them to a list.
    """
    if isinstance(shap_values, list):
        return shap_values
    arr = np.asarray(shap_values)
    if arr.ndim == 3:  # (samples, features, classes)
        return [arr[:, :, c] for c in range(arr.shape[2])]
    return [arr]  # single-output model (e.g. binary xgboost)


def to_base_list(expected_value, n_classes: int) -> list:
    """Normalise the explainer base value into a per-class list."""
    if np.ndim(expected_value) == 0:
        return [float(expected_value)] * n_classes
    return list(np.asarray(expected_value).ravel())


def save(fig_name: str) -> None:
    """Save the current matplotlib figure to the figures directory."""
    path = config.FIGURES_DIR / fig_name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path.name}")


def main() -> None:
    """Compute Shapley values and write every required plot."""
    parser = argparse.ArgumentParser(description="SHAP explanations.")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "xgboost"],
        help="Tree model to explain (default: random_forest).",
    )
    parser.add_argument("--sample", type=int, default=500)
    args = parser.parse_args()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = load_tree_model(args.model)
    x_test = pd.read_parquet(config.X_TEST)
    # SHAP on the full test set is slow; a sample is plenty for plots.
    x = x_test.sample(
        min(args.sample, len(x_test)), random_state=config.RANDOM_STATE
    ).reset_index(drop=True)

    # TreeExplainer + Shapley values --------------------------------------
    explainer = shap.TreeExplainer(model)
    sv_classes = to_class_list(explainer.shap_values(x))
    n_classes = len(sv_classes)
    base_values = to_base_list(explainer.expected_value, n_classes)
    churn_idx = CHURN_CLASS if n_classes > 1 else 0
    sv_churn = sv_classes[churn_idx]
    base_churn = base_values[churn_idx]
    print(f"Computed SHAP values: {n_classes} class output(s)")

    # Modern Explanation object for the churn class (drives several plots).
    exp_churn = shap.Explanation(
        values=sv_churn,
        base_values=np.repeat(base_churn, len(x)),
        data=x.values,
        feature_names=list(x.columns),
    )

    # 1. Summary plot for each class --------------------------------------
    for c in range(n_classes):
        shap.summary_plot(sv_classes[c], x, show=False)
        plt.title(f"SHAP summary - class {c}")
        save(f"summary_class_{c}.png")

    # 2. All-points explanation: beeswarm ---------------------------------
    shap.plots.beeswarm(exp_churn, show=False)
    save("beeswarm_all_points.png")

    # 3. Mean SHAP (global importance) bar plot ---------------------------
    shap.plots.bar(exp_churn, show=False)
    save("mean_shap_bar.png")

    # 4. All-points interactive force plot (saved as HTML) ----------------
    force_all = shap.force_plot(base_churn, sv_churn, x)
    shap.save_html(str(config.FIGURES_DIR / "force_all_points.html"), force_all)
    print("  wrote force_all_points.html")

    # 5. Single-point explanation: waterfall + force ----------------------
    shap.plots.waterfall(exp_churn[0], show=False)
    save("waterfall_point0.png")

    shap.force_plot(base_churn, sv_churn[0], x.iloc[0], matplotlib=True, show=False)
    save("force_point0.png")

    # 6. Dependence plot for the most important feature -------------------
    top_feature = x.columns[np.argsort(np.abs(sv_churn).mean(0))[-1]]
    shap.dependence_plot(top_feature, sv_churn, x, show=False)
    save(f"dependence_{top_feature}.png")

    print(f"All SHAP figures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
