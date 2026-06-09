from dagster import (
    asset,
    AssetIn,
    AssetKey,
)

from sklearn.ensemble import RandomForestRegressor

import joblib
import os


def train_random_forest(X_train, y_train, params=None):

    params = params or {}

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        **params,
    )

    model.fit(X_train, y_train)

    return model


def create_random_forest_asset(
    experiment_name: str,
    model_name: str,
    cfg: dict,
):

    model_params = cfg.get("params", {})

    asset_name = f"{experiment_name}_{model_name}_model"

    @asset(
        name=asset_name,
        group_name="model_training",
        ins={
            "X_train": AssetIn(
                key=AssetKey([
                    f"{experiment_name}_{model_name}_X_train_processed"
                ])
            ),
            "y_train": AssetIn(
                key=AssetKey([
                    f"{experiment_name}_{model_name}_y_train_processed"
                ])
            ),
        },
    )
    def model_asset(context, X_train, y_train):

        model = train_random_forest(
            X_train,
            y_train,
            params=model_params,
        )

        path = (
            f"data/output_data/models/"
            f"{experiment_name}_{model_name}.pkl"
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump(
            {
                "model": model,
                "features": list(X_train.columns),
                "model_type": model_name,
            },
            path,
        )

        context.add_output_metadata({
            "model_type": "RandomForest",
            "experiment": experiment_name,
            "model": model_name,
            "rows": X_train.shape[0],
            "columns": X_train.shape[1],
            "feature_count": len(X_train.columns),
            "model_path": path,
        })

        return model

    return model_asset