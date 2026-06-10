from dagster import asset, AssetIn, AssetKey, multi_asset, AssetOut

from bike_rental.utils.model_specific_preprocessing import (
    validate_feature_alignment,
)
import pandas as pd
from bike_rental.utils.fit_transform_helper import (
    fit_transform_features,
    transform_features,
    get_fitted_preprocessor,
)

from bike_rental.utils.data_asset_creation_helper import _get_selected_features, _base_metadata, _prefix, _ak

def create_ml_ready_assets(
    experiment_name: str,
    model_name: str,
    experiment_cfg: dict,
    model_cfg: dict,
):

    selected_features = _get_selected_features(model_cfg)
    target = experiment_cfg["target"]
    preprocessing_cfg = model_cfg["preprocessing"]

    prefix = _prefix(experiment_name, model_name)

    dataset_name = experiment_cfg["dataset"]
    train_dataset_key = AssetKey([f"train_{dataset_name}"])
    test_dataset_key = AssetKey([f"test_{dataset_name}"])

    # =========================================================
    # RAW (ONLY SOURCE OF TRUTH)
    # =========================================================

    @multi_asset(
        name=f"{prefix}_raw",
        group_name=experiment_name,
        outs={
            f"{prefix}_X_train": AssetOut(io_manager_key="csv_io_manager"),
            f"{prefix}_y_train": AssetOut(io_manager_key="csv_io_manager"),
            f"{prefix}_X_test": AssetOut(io_manager_key="csv_io_manager"),
            f"{prefix}_y_test": AssetOut(io_manager_key="csv_io_manager"),
        },
        ins={
            "train_dataset": AssetIn(key=train_dataset_key),
            "test_dataset": AssetIn(key=test_dataset_key),
        },
    )
    def raw_assets(context, train_dataset, test_dataset):
        validate_feature_alignment(train_dataset, selected_features, model_cfg)

        X_train = train_dataset[selected_features].copy()
        y_train = train_dataset[target].copy()

        X_test = test_dataset[selected_features].copy()
        y_test = test_dataset[target].copy()

        return X_train, y_train, X_test, y_test

    # =========================================================
    # X TRAIN PROCESSED
    # =========================================================

    @asset(
        name=f"{prefix}_X_train_processed",
        ins={
            "X_train": AssetIn(key=AssetKey([f"{prefix}_X_train"]))
        },
        group_name=experiment_name,
        io_manager_key="csv_io_manager",
    )
    def X_train_processed(context, X_train):

        X_processed, _ = fit_transform_features(X_train, model_cfg)

        context.add_output_metadata({
            **_base_metadata(experiment_name, model_name, experiment_cfg),
            "rows": X_processed.shape[0],
            "columns": X_processed.shape[1],
            "scale_numeric": preprocessing_cfg["scale_numeric"],
            "one_hot_encode": preprocessing_cfg["one_hot_encode"],
        })

        return X_processed

    # =========================================================
    # X TEST PROCESSED
    # =========================================================

    @asset(
        name=f"{prefix}_X_test_processed",
        ins={
            "X_test": AssetIn(key=AssetKey([f"{prefix}_X_test"])),
            "X_train": AssetIn(key=AssetKey([f"{prefix}_X_train"])),
        },
        group_name=experiment_name,
        io_manager_key="csv_io_manager",
    )
    def X_test_processed(context, X_test, X_train):

        preprocessor = get_fitted_preprocessor(X_train, model_cfg)

        X_processed = transform_features(X_test, preprocessor)

        context.add_output_metadata({
            **_base_metadata(experiment_name, model_name, experiment_cfg),
            "rows": X_processed.shape[0],
            "columns": X_processed.shape[1],
            "scale_numeric": preprocessing_cfg["scale_numeric"],
            "one_hot_encode": preprocessing_cfg["one_hot_encode"],
        })

        return X_processed

    # =========================================================
    # y TRAIN PROCESSED
    # =========================================================

    @asset(
        name=f"{prefix}_y_train_processed",
        ins={
            "y_train": AssetIn(key=AssetKey([f"{prefix}_y_train"]))
        },
        group_name=experiment_name,
        io_manager_key="csv_io_manager",
    )
    def y_train_processed(context, y_train):

        context.add_output_metadata({
            **_base_metadata(experiment_name, model_name, experiment_cfg),
            "rows": len(y_train),
            "mean": float(y_train.mean()),
            "min": float(y_train.min()),
            "max": float(y_train.max()),
        })

        return y_train.copy()

    # =========================================================
    # y TEST PROCESSED
    # =========================================================

    @asset(
        name=f"{prefix}_y_test_processed",
        ins={
            "y_test": AssetIn(key=AssetKey([f"{prefix}_y_test"]))
        },
        group_name=experiment_name,
        io_manager_key="csv_io_manager",
    )
    def y_test_processed(context, y_test):

        context.add_output_metadata({
            **_base_metadata(experiment_name, model_name, experiment_cfg),
            "rows": len(y_test),
            "mean": float(y_test.mean()),
            "min": float(y_test.min()),
            "max": float(y_test.max()),
        })

        return y_test.copy()

    return [
        raw_assets,
        X_train_processed,
        X_test_processed,
        y_train_processed,
        y_test_processed,
    ]