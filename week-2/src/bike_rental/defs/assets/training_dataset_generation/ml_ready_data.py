from dagster import asset, AssetIn, AssetKey, multi_asset, AssetOut

from bike_rental.utils.model_specific_preprocessing import validate_feature_alignment
from bike_rental.utils.fit_transform_helper import fit_transform_features, transform_features

from bike_rental.utils.data_asset_creation_helper import _get_selected_features, _base_metadata, _prefix

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

    @asset(
        name=f"{prefix}_ml_ready_dataset",
        group_name=experiment_name,
        ins={
            "train_dataset": AssetIn(key=AssetKey(["train_dataset_hourly"])),
            "test_dataset": AssetIn(key=AssetKey(["test_dataset_hourly"])),
        },
    )
    def ml_ready_dataset(context, train_dataset, test_dataset):

        # -----------------------------------
        # Validate feature alignment
        # -----------------------------------
        validate_feature_alignment(
            train_dataset,
            selected_features,
            model_cfg
        )

        # -----------------------------------
        # Split X/y using config
        # -----------------------------------
        X_train = train_dataset[selected_features].copy()
        y_train = train_dataset[target].copy()

        X_test = test_dataset[selected_features].copy()
        y_test = test_dataset[target].copy()

        # -----------------------------------
        # Fit preprocessing ONLY on train
        # -----------------------------------
        X_train_processed, preprocessor = fit_transform_features(
            X_train,
            model_cfg
        )

        # -----------------------------------
        # Apply SAME preprocessing to test
        # -----------------------------------
        X_test_processed = transform_features(
            X_test,
            preprocessor
        )

        # -----------------------------------
        # Metadata (merged from all old assets)
        # -----------------------------------
        context.add_output_metadata({
            **_base_metadata(experiment_name, model_name, experiment_cfg),

            "train_rows": len(X_train_processed),
            "test_rows": len(X_test_processed),
            "n_features": X_train_processed.shape[1],

            "selected_features": selected_features,

            "scale_numeric": preprocessing_cfg["scale_numeric"],
            "one_hot_encode": preprocessing_cfg["one_hot_encode"],
            "impute_numeric": preprocessing_cfg["impute_numeric"],
            "impute_categorical": preprocessing_cfg["impute_categorical"],
            "remove_low_variance": preprocessing_cfg["remove_low_variance"],
        })

        # -----------------------------------
        # SINGLE RETURN OBJECT (clean boundary)
        # -----------------------------------
        return {
            "X_train": X_train_processed,
            "y_train": y_train,
            "X_test": X_test_processed,
            "y_test": y_test,
            "feature_names": list(X_train_processed.columns),
        }

    return [ml_ready_dataset]