from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)

from sklearn.impute import SimpleImputer

def build_preprocessor(config):

    features = config["features"]
    preprocessing = config["preprocessing"]
    numeric = features["numeric"]
    categorical = features["categorical"]
    transformers = []


    numeric_steps = []

    if preprocessing["impute_numeric"]:

        numeric_steps.append(
            (
                "imputer",
                SimpleImputer(
                    strategy=preprocessing[
                        "impute_numeric"
                    ]
                ),
            )
        )

    if preprocessing["scale_numeric"]:

        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    if numeric_steps:

        transformers.append(
            (
                "numeric",
                Pipeline(numeric_steps),
                numeric,
            )
        )



    categorical_steps = []

    if preprocessing["impute_categorical"]:

        categorical_steps.append(
            (
                "imputer",
                SimpleImputer(
                    strategy=preprocessing[
                        "impute_categorical"
                    ]
                ),
            )
        )

    if preprocessing["one_hot_encode"]:

        categorical_steps.append(
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            )
        )

    if categorical_steps:

        transformers.append(
            (
                "categorical",
                Pipeline(categorical_steps),
                categorical,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
    )