from dagster import asset, AssetIn, AssetKey
import joblib
import os

from bike_rental.defs.assets.models.estimator_registry import MODEL_REGISTRY
from bike_rental.defs.assets.evaluation.eval_metrics import evaluate_model


def create_model_asset(
    experiment_name: str,
    model_name: str,
    estimator_name: str,
):
    """
    Generic model training + evaluation asset.
    """

    asset_name = f"{experiment_name}_{model_name}_model"

    ml_ready_key = AssetKey(
        [f"{experiment_name}_{model_name}_ml_ready_dataset"]
    )

    @asset(
        name=asset_name,
        group_name="model_training",
        ins={
            "ml_ready": AssetIn(key=ml_ready_key),
        },
    )
    def model_asset(context, ml_ready):

        # -------------------------
        # Get estimator
        # -------------------------
        estimator_fn = MODEL_REGISTRY[estimator_name]
        model = estimator_fn()

        # -------------------------
        # Data unpacking
        # -------------------------
        X_train = ml_ready["X_train"]
        y_train = ml_ready["y_train"]
        X_test = ml_ready["X_test"]
        y_test = ml_ready["y_test"]

        # -------------------------
        # Train
        # -------------------------
        model.fit(X_train, y_train)

        # -------------------------
        # Evaluate
        # -------------------------
        metrics = evaluate_model(model, X_test, y_test)

        # -------------------------
        # Persist model
        # -------------------------
        path = (
            f"data/output_data/models/"
            f"{experiment_name}_{model_name}.pkl"
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump(
            {
                "model": model,
                "features": list(X_train.columns),
                "metrics": metrics,
            },
            path,
        )

        context.add_output_metadata({
            "model_type": estimator_name,
            "experiment": experiment_name,
            "model": model_name,
            "model_path": path,
            **metrics,
        })

        return {
            "model": model,
            "metrics": metrics,
        }

    return model_asset