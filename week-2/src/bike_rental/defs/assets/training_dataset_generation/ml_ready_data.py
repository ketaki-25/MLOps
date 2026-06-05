from dagster import AssetExecutionContext
from dagster import asset
from bike_rental.utils.feature_selection_helper import select_input_features, select_target_features
from bike_rental.utils.model_specific_preprocessing import validate_feature_alignment
from bike_rental.utils.fit_transform_helper import fit_transform_features, transform_features, get_fitted_preprocessor
import pandas as pd
import numpy as np

@asset(group_name="input_target_feature_split", required_resource_keys={"experiment_config"}, io_manager_key="csv_io_manager")
def X_train_hourly(context: AssetExecutionContext, train_dataset_hourly):
    """Create training_dataset_generation feature matrix using the active experiment configuration."""

    selected_features, cfg = select_input_features(context)

    validate_feature_alignment(
        train_dataset_hourly,
        selected_features,
        cfg,
    )

    X = train_dataset_hourly[
        selected_features
    ].copy()

    context.add_output_metadata(
        {
            "rows": len(X),
            "columns": X.shape[1],
            "selected_features": selected_features,
            "model": cfg["model"],
            "target": cfg["target"],
        }
    )

    return X


@asset(group_name="input_target_feature_split", required_resource_keys={"experiment_config"}, io_manager_key="csv_io_manager")
def y_train_hourly(context: AssetExecutionContext, train_dataset_hourly):
    """Create training_dataset_generation target vector using the active experiment configuration."""

    target_col, cfg = (select_target_features(context))

    y = train_dataset_hourly[target_col].copy()

    context.add_output_metadata(
        {
            "target_column": target_col,
            "rows": len(y),
            "mean": float(y.mean()),
            "min": float(y.min()),
            "max": float(y.max()),
            "null_values": int(y.isna().sum()),
        }
    )

    return y

@asset(group_name="input_target_feature_split", required_resource_keys={"experiment_config"}, io_manager_key="csv_io_manager")
def X_test_hourly(
    context: AssetExecutionContext,
    test_dataset_hourly,
):
    """
    Create test feature matrix using the active
    experiment configuration.
    """
    selected_features, cfg = select_input_features(context)

    X = test_dataset_hourly[
        selected_features
    ].copy()

    context.add_output_metadata(
        {
            "rows": len(X),
            "columns": X.shape[1],
            "selected_features": selected_features,
            "model": cfg["model"],
            "target": cfg["target"],
        }
    )

    return X


@asset(group_name="input_target_feature_split", required_resource_keys={"experiment_config"}, io_manager_key="csv_io_manager")
def y_test_hourly(
    context: AssetExecutionContext,
    test_dataset_hourly,
):
    """Create test target vector for model evaluation."""

    target_col, cfg = (select_target_features(context))

    y = test_dataset_hourly[target_col].copy()

    context.add_output_metadata(
        {
            "target_column": target_col,
            "rows": len(y),
            "mean": float(y.mean()),
            "min": float(y.min()),
            "max": float(y.max()),
            "null_values": int(y.isna().sum()),
        }
    )

    return y

@asset(group_name="model_specific_preprocessing", required_resource_keys={"experiment_config"}, io_manager_key="csv_io_manager")
def X_train_processed_hourly(
    context: AssetExecutionContext,
    X_train_hourly: pd.DataFrame,
):
    """Apply configured preprocessing pipeline to training_dataset_generation features.
    Fits preprocessing on training_dataset_generation data only and returns transformed training_dataset_generation matrix."""

    cfg = (
        context.resources
        .experiment_config
        .get_active_experiment()
    )

    (
        X_train_processed,
        preprocessor,
    ) = fit_transform_features(
        X_train_hourly,
        cfg,
    )



    context.add_output_metadata(
        {
            "rows": X_train_processed.shape[0],
            "columns": X_train_processed.shape[1],
            "model": cfg["model"],
            "scale_numeric": cfg["preprocessing"][
                "scale_numeric"
            ],
            "one_hot_encode": cfg["preprocessing"][
                "one_hot_encode"
            ],
        }
    )

    assert not X_train_processed.isna().any().any(), "NaNs in training data"
    #assert not np.isinf(X_train_processed.values).any(), "Inf in training data"

    return X_train_processed

#TODO:
# Do not fit twice reimplement the design to incorporate the same fitted preprocessor

@asset(
    group_name="model_specific_preprocessing",
    required_resource_keys={"experiment_config"}, io_manager_key="csv_io_manager"
)
def X_test_processed_hourly(
    context: AssetExecutionContext,
    X_test_hourly: pd.DataFrame,
    X_train_hourly: pd.DataFrame,
):
    """
    Apply configured preprocessing pipeline
    to test features.

    Uses preprocessor fitted on training_dataset_generation data.
    """
    cfg = (
        context.resources
        .experiment_config
        .get_active_experiment()
    )

    preprocessor = get_fitted_preprocessor(X_train_hourly, cfg=cfg)


    X_test_processed = transform_features(
        X_test_hourly,
        preprocessor,
    )

    context.add_output_metadata(
        {
            "rows": X_test_processed.shape[0],
            "columns": X_test_processed.shape[1],
            "model": cfg["model"],
            "scale_numeric": cfg["preprocessing"][
                "scale_numeric"
            ],
            "one_hot_encode": cfg["preprocessing"][
                "one_hot_encode"
            ],
        }
    )

    assert not X_test_processed.isna().any().any(), "NaNs in training data"
    #assert not np.isinf(X_test_processed.values).any(), "Inf in training data"

    return X_test_processed

@asset(
    group_name="model_specific_preprocessing",
    required_resource_keys={"experiment_config"}, io_manager_key="csv_io_manager"
)
def y_train_processed_hourly(
    context: AssetExecutionContext,
    y_train_hourly,
):
    """
    Return processed training_dataset_generation target.

    Currently no target transformations
    are applied.
    """

    y = y_train_hourly.copy()

    context.add_output_metadata(
        {
            "rows": len(y),
            "mean": float(y.mean()),
            "min": float(y.min()),
            "max": float(y.max()),
        }
    )

    assert not y.isna().any().any(), "NaNs in training data"
    #assert not np.isinf(y.values).any(), "Inf in training data"

    return y

@asset(
    group_name="model_specific_preprocessing",
    required_resource_keys={"experiment_config"}, io_manager_key="csv_io_manager"
)
def y_test_processed_hourly(
    context: AssetExecutionContext,
    y_test_hourly,
):
    """Return processed test target.

    Currently no target transformations are applied."""

    y = y_test_hourly.copy()

    context.add_output_metadata(
        {
            "rows": len(y),
            "mean": float(y.mean()),
            "min": float(y.min()),
            "max": float(y.max()),
        }
    )

    assert not y.isna().any().any(), "NaNs in training data"
    #assert not np.isinf(y.values).any(), "Inf in training data"

    return y