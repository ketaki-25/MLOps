from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


VALID_NUMERIC_IMPUTERS = {"mean", "median", "most_frequent", "constant"}
VALID_CATEGORICAL_IMPUTERS = {"most_frequent", "constant"}


def _validate_feature_group(features, group_name):
    value = features.get(group_name, [])

    if value is None:
        return []

    if not isinstance(value, list):
        raise TypeError(
            f"features['{group_name}'] must be a list. "
            f"Received {type(value).__name__}."
        )
    return value


def _validate_bool(preprocessing, key):
    value = preprocessing.get(key)

    if not isinstance(value, bool):
        raise TypeError(
            f"preprocessing['{key}'] must be bool. "
            f"Received {type(value).__name__}."
        )
    return value


def validate_columns_exist(X, config):
    """prevents sklearn cryptic errors"""

    features = config["features"]

    configured = []
    for group, cols in features.items():
        if isinstance(cols, list):
            configured.extend(cols)

    missing = sorted(set(configured) - set(X.columns))

    if missing:
        raise ValueError(
            "\nMissing columns in dataframe:\n"
            f"{missing}\n\n"
            "Available columns:\n"
            f"{sorted(X.columns.tolist())}\n"
        )

def validate_feature_alignment(X, selected_features, cfg):

    missing = set(selected_features) - set(X.columns)

    if missing:
        raise ValueError(
            f"Selected features not in dataframe: {missing}"
        )


def build_preprocessor(config, X=None):
    """
    X is optional but HIGHLY recommended for validation
    """

    if "features" not in config or "preprocessing" not in config:
        raise ValueError("Missing 'features' or 'preprocessing' section.")

    features = config["features"]
    preprocessing = config["preprocessing"]

    numeric = _validate_feature_group(features, "numeric")
    categorical = _validate_feature_group(features, "categorical")
    boolean = _validate_feature_group(features, "boolean")
    cyclic = _validate_feature_group(features, "cyclic")
    id_columns = _validate_feature_group(features, "id_columns")

    scale_numeric = _validate_bool(preprocessing, "scale_numeric")
    one_hot_encode = _validate_bool(preprocessing, "one_hot_encode")

    impute_numeric = preprocessing.get("impute_numeric")
    impute_categorical = preprocessing.get("impute_categorical")

    if X is not None:
        validate_columns_exist(X, config)

    transformers = []

    # ---------------- NUMERIC ----------------
    numeric_steps = []

    if impute_numeric is not None:
        numeric_steps.append(
            ("imputer", SimpleImputer(strategy=impute_numeric))
        )

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(numeric_steps) if numeric_steps else "passthrough",
                numeric,
            )
        )

    # ---------------- CATEGORICAL ----------------
    categorical_steps = []

    if impute_categorical is not None:
        categorical_steps.append(
            ("imputer", SimpleImputer(strategy=impute_categorical))
        )

    if one_hot_encode:
        categorical_steps.append(
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(categorical_steps) if categorical_steps else "passthrough",
                categorical,
            )
        )

    if boolean:
        transformers.append(("boolean", "passthrough", boolean))

    if cyclic:
        transformers.append(("cyclic", "passthrough", cyclic))

    if id_columns:
        transformers.append(("id_columns", "passthrough", id_columns))

    if not transformers:
        raise ValueError("No feature groups defined.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )