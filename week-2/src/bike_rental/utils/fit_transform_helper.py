from bike_rental.utils.model_specific_preprocessing import build_preprocessor
import pandas as pd

def fit_transform_features(
    x_train,
    cfg,
):
    """
    Fit preprocessing pipeline on training_dataset_generation data and
    return transformed training_dataset_generation features plus the
    fitted preprocessor.
    """

    preprocessor = build_preprocessor(cfg)

    X_train_processed = (
        preprocessor.fit_transform(x_train)
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=x_train.index,
    )

    return (
        X_train_processed,
        preprocessor,
    )

def transform_features(
    x_test,
    preprocessor,
):
    """
    Transform a dataset using an already fitted
    preprocessor and return a DataFrame with
    feature names preserved.
    """

    X_processed = (
        preprocessor.transform(x_test)
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    X_processed = pd.DataFrame(
        X_processed,
        columns=feature_names,
        index=x_test.index,
    )

    return X_processed

def get_fitted_preprocessor(
    X_train: pd.DataFrame,
    cfg: dict,
):
    """
    Build and fit the preprocessing pipeline
    using training_dataset_generation data only.
    """

    preprocessor = build_preprocessor(cfg)

    preprocessor.fit(X_train)

    return preprocessor
