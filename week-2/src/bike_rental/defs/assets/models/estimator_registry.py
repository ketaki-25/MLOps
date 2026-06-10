from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def linear_regression():
    return LinearRegression()


def random_forest():
    return RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )


def xgboost():
    return XGBRegressor(
        enable_categorical=True,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )


MODEL_REGISTRY = {
    "linear_regression": linear_regression,
    "random_forest": random_forest,
    "xgboost": xgboost,
}
