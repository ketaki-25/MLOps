from dagster import (
    asset,
    AssetIn,
    AssetKey,
)

from sklearn.linear_model import LinearRegression

import joblib
import os


def train_linear_model(X_train, y_train):

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    return model


def create_linear_regression_asset(
    experiment_name: str,
    model_name: str,
    cfg: dict,
):
    """
    Linear regression training asset factory.

    NOTE:
    - model_name and cfg are included for interface consistency
      across all model factories.
    - cfg is not used yet but kept for extensibility.
    """

    asset_name = f"{experiment_name}_{model_name}_model"

    @asset(
        name=asset_name,
        group_name="model_training",
        ins={
            "X_train": AssetIn(
                key=AssetKey(
                    [f"{experiment_name}_{model_name}_X_train_processed"]
                )
            ),
            "y_train": AssetIn(
                key=AssetKey(
                    [f"{experiment_name}_{model_name}_y_train_processed"]
                )
            ),
        },
    )
    def model_asset(
        context,
        X_train,
        y_train,
    ):

        model = train_linear_model(
            X_train,
            y_train,
        )

        path = (
            f"data/output_data/models/"
            f"{experiment_name}_{model_name}.pkl"
        )

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        joblib.dump(
            {
                "model": model,
                "features": list(X_train.columns),
            },
            path,
        )

        context.add_output_metadata(
            {
                "model_type": "LinearRegression",
                "experiment": experiment_name,
                "model": model_name,
                "model_path": path,
            }
        )

        return model

    return model_asset