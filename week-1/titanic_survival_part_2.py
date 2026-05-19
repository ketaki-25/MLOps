import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

data = pd.read_csv("../data/titanic.csv")
data.head()

FEATURES = ["sex", "age", "family_size", "fare", "1st_class", "2nd_class", "3rd_class"]
TARGET = "survived"

features = data[FEATURES]

target = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

# TODO: Implement feature standardization.

'''
scaler = StandardScaler()

# Fit ONLY on training data
x_train_standardized = scaler.fit_transform(X_train)

# Use same statistics on test data
x_test_standardized = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples, {X_train_scaled}")
print(f"Test set: {X_test.shape[0]} samples")
'''

numeric_features = ["age", "family_size", "fare"]
binary_features = ["sex", "1st_class", "2nd_class", "3rd_class"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
    ],
    remainder="passthrough"
)

x_train_standardized = preprocessor.fit_transform(X_train)
x_test_standardized = preprocessor.transform(X_test)

print(x_train_standardized, x_test_standardized)
'''

def train_standardizer(features_dataset):

    x_train_standardized = features_dataset.copy()
    for column in features_dataset.columns:
        mean = features_dataset[column].mean()
        std = features_dataset[column].std()

        print(f"Mean of {column}: {mean:.2f}, standard deviation of {column} {std:.2f}")

        x_train_standardized[column] = (features_dataset[column] - mean) / std


    return x_train_standardized

def test_standardizer(train_features_dataset, test_features_dataset):

    x_test_standardized = test_features_dataset.copy()
    for column in train_features_dataset.columns:
        mean = train_features_dataset[column].mean()
        std = train_features_dataset[column].std()

        print(f"Mean of {column}: {mean:.2f}, standard deviation of {column} {std:.2f}")

        x_test_standardized[column] = (test_features_dataset[column] - mean) / std


    return x_test_standardized

x_train_standardized = train_standardizer(X_train)
x_test_standardized = test_standardizer(X_train, X_test)

print(x_train_standardized, x_test_standardized)
'''


def sigmoid(z: np.ndarray) -> np.ndarray:
    """A numerically stable sigmoid function."""
    # TODO: Implement this function.
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )


print(sigmoid(x_test_standardized))


def binary_cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Compute the mean binary cross-entropy loss.

    Take care in your implementation to ensure that the cross entropy is always positive,
    and that it stays stable for very small probabilities (y_hat \approx 0).
    """
    # TODO: Implement this function.


def logistic_regression_gd(
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
) -> tuple[np.ndarray, float, list[float]]:
    """Train logistic regression via gradient descent.

    Returns (weights, bias, loss_history).
    """
    # TODO: Implement this function.


# Train our model on the standardized features
w, b, loss_history = logistic_regression_gd(X_train_s, y_train, lr=0.1, max_iter=1000)

print(f"Final loss: {loss_history[-1]:.6f}")
print(f"Iterations: {len(loss_history)}")
print(f"\nLearned weights:")
for name, weight in zip(FEATURES, w):
    print(f"  {name:>30s}: {weight:+.4f}")
print(f"  {'bias':>30s}: {b:+.4f}")


