from dagster import (
    asset,
    AssetIn,
    AssetKey,
)

import pandas as pd

from bike_rental.utils.model_specific_preprocessing import (
    validate_feature_alignment,
)

from bike_rental.utils.fit_transform_helper import (
    fit_transform_features,
    transform_features,
    get_fitted_preprocessor,
)


def _get_selected_features(model_cfg: dict) -> list[str]:
    """
    Flatten all configured feature groups into a single feature list.
    """

    selected_features = []

    for feature_group in model_cfg["features"].values():
        if feature_group:
            selected_features.extend(feature_group)

    return selected_features


def _base_metadata(
    experiment_name: str,
    model_name: str,
    experiment_cfg: dict,
) -> dict:
    """
    Common metadata attached to all assets.
    """

    return {
        "experiment": experiment_name,
        "model": model_name,
        "target": experiment_cfg["target"],
        "dataset": experiment_cfg["dataset"],
    }


def create_ml_ready_assets(
    experiment_name: str,
    model_name: str,
    experiment_cfg: dict,
    model_cfg: dict,
):
    """
    Create all ML-ready assets for a single model within an experiment.

    Example generated assets:

        total_hourly_linear_regression_X_train
        total_hourly_linear_regression_X_test
        total_hourly_linear_regression_X_train_processed
        total_hourly_linear_regression_X_test_processed
        total_hourly_linear_regression_y_train
        total_hourly_linear_regression_y_test
    """

    selected_features = _get_selected_features(model_cfg)

    target = experiment_cfg["target"]

    preprocessing_cfg = model_cfg["preprocessing"]

    asset_prefix = f"{experiment_name}_{model_name}"

    train_dataset_key = AssetKey(["train_dataset_hourly"])
    test_dataset_key = AssetKey(["test_dataset_hourly"])

    # =======================================================
    # X TRAIN
    # =======================================================

    @asset(
        name=f"{asset_prefix}_X_train",
        ins={
            "dataset": AssetIn(
                key=train_dataset_key
            )
        },
        group_name=experiment_name,
        io_manager_key="csv_io_manager",
    )
    def X_train(
        context,
        dataset: pd.DataFrame,
    ):

        validate_feature_alignment(
            dataset,
            selected_features,
            model_cfg,
        )

        X = dataset[selected_features].copy()

        context.add_output_metadata({
            **_base_metadata(
                experiment_name,
                model_name,
                experiment_cfg,
            ),
            "rows": X.shape[0],
            "columns": X.shape[1],
            "feature_count": len(selected_features),
            "selected_features": selected_features,
        })

        return X

    # =======================================================
    # y TRAIN
    # =======================================================

    @asset(
        name=f"{asset_prefix}_y_train",
        ins={
            "dataset": AssetIn(
                key=train_dataset_key
            )
        },
        group_name=experiment_name,
        io_manager_key="csv_io_manager",
    )
    def y_train(
        context,
        dataset: pd.DataFrame,
    ):

        y = dataset[target].copy()

        context.add_output_metadata({
            **_base_metadata(
                experiment_name,
                model_name,
                experiment_cfg,
            ),
            "rows": len(y),
            "null_values": int(y.isna().sum()),
        })

        return y

    # =======================================================
    # X TEST
    # =======================================================

    @asset(
        name=f"{asset_prefix}_X_test",
        ins={
            "dataset": AssetIn(
                key=test_dataset_key
            )
        },
        group_name=experiment_name,
        io_manager_key="csv_io_manager",
    )
    def X_test(
        context,
        dataset: pd.DataFrame,
    ):

        X = dataset[selected_features].copy()

        context.add_output_metadata({
            **_base_metadata(
                experiment_name,
                model_name,
                experiment_cfg,
            ),
            "rows": X.shape[0],
            "columns": X.shape[1],
            "feature_count": len(selected_features),
            "selected_features": selected_features,
        })

        return X

    # =======================================================
    # y TEST
    # =======================================================

    @asset(
        name=f"{asset_prefix}_y_test",
        ins={
            "dataset": AssetIn(
                key=test_dataset_key
            )
        },
        group_name=experiment_name,
        io_manager_key="csv_io_manager",
    )
    def y_test(
        context,
        dataset: pd.DataFrame,
    ):

        y = dataset[target].copy()

        context.add_output_metadata({
            **_base_metadata(
                experiment_name,
                model_name,
                experiment_cfg,
            ),
            "rows": len(y),
            "null_values": int(y.isna().sum()),
        })

        return y

    # =======================================================
    # X TRAIN PROCESSED
    # =======================================================

    @asset(
        name=f"{asset_prefix}_X_train_processed",
        ins={
            "X_train": AssetIn(
                key=AssetKey(
                    [f"{asset_prefix}_X_train"]
                )
            )
        },
        group_name=experiment_name,
        io_manager_key="csv_io_manager",
    )
    def X_train_processed(
        context,
        X_train: pd.DataFrame,
    ):

        X_processed, _ = fit_transform_features(
            X_train,
            model_cfg,
        )

        context.add_output_metadata({
            **_base_metadata(
                experiment_name,
                model_name,
                experiment_cfg,
            ),
            "rows": X_processed.shape[0],
            "columns": X_processed.shape[1],
            "scale_numeric": preprocessing_cfg["scale_numeric"],
            "one_hot_encode": preprocessing_cfg["one_hot_encode"],
        })

        return X_processed

    # =======================================================
    # X TEST PROCESSED
    # =======================================================

    @asset(
        name=f"{asset_prefix}_X_test_processed",
        ins={
            "X_test": AssetIn(
                key=AssetKey(
                    [f"{asset_prefix}_X_test"]
                )
            ),
            "X_train": AssetIn(
                key=AssetKey(
                    [f"{asset_prefix}_X_train"]
                )
            ),
        },
        group_name=experiment_name,
        io_manager_key="csv_io_manager",
    )
    def X_test_processed(
        context,
        X_test: pd.DataFrame,
        X_train: pd.DataFrame,
    ):

        preprocessor = get_fitted_preprocessor(
            X_train,
            model_cfg,
        )

        X_processed = transform_features(
            X_test,
            preprocessor,
        )

        context.add_output_metadata({
            **_base_metadata(
                experiment_name,
                model_name,
                experiment_cfg,
            ),
            "rows": X_processed.shape[0],
            "columns": X_processed.shape[1],
            "scale_numeric": preprocessing_cfg["scale_numeric"],
            "one_hot_encode": preprocessing_cfg["one_hot_encode"],
        })

        return X_processed

    # =======================================================
    # y TRAIN PROCESSED
    # =======================================================

    @asset(
        name=f"{asset_prefix}_y_train_processed",
        ins={
            "y_train": AssetIn(
                key=AssetKey(
                    [f"{asset_prefix}_y_train"]
                )
            )
        },
        group_name=experiment_name,
        io_manager_key="csv_io_manager",
    )
    def y_train_processed(
        context,
        y_train,
    ):

        context.add_output_metadata({
            **_base_metadata(
                experiment_name,
                model_name,
                experiment_cfg,
            ),
            "rows": len(y_train),
            "mean": float(y_train.mean()),
            "min": float(y_train.min()),
            "max": float(y_train.max()),
        })

        return y_train.copy()

    # =======================================================
    # y TEST PROCESSED
    # =======================================================

    @asset(
        name=f"{asset_prefix}_y_test_processed",
        ins={
            "y_test": AssetIn(
                key=AssetKey(
                    [f"{asset_prefix}_y_test"]
                )
            )
        },
        group_name=experiment_name,
        io_manager_key="csv_io_manager",
    )
    def y_test_processed(
        context,
        y_test,
    ):

        context.add_output_metadata({
            **_base_metadata(
                experiment_name,
                model_name,
                experiment_cfg,
            ),
            "rows": len(y_test),
            "mean": float(y_test.mean()),
            "min": float(y_test.min()),
            "max": float(y_test.max()),
        })

        return y_test.copy()

    return [
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_processed,
        X_test_processed,
        y_train_processed,
        y_test_processed,
    ]