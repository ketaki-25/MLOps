import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------
df = pd.read_csv("data/titanic.csv")

print(df.describe())
print(df.isnull().sum())

# ---------------------------------------------------------------
# 2. DATA EXPLORATION
# ---------------------------------------------------------------
numeric_features = ["age", "family_size", "fare"]
binary_features = ["sex", "1st_class", "2nd_class", "3rd_class"]

for feature in df.columns:
    if feature in binary_features:
        sns.barplot(x=feature, y="survived", data=df)
        plt.title(f"Survival Rate by {feature}")
        plt.show()

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Age group by survivors
viz_df = df.copy()
bins = np.arange(0, viz_df["age"].max() + 5, 5)

viz_df["age_group"] = pd.cut(viz_df["age"], bins=bins)

plt.figure(figsize=(12, 6))

sns.countplot(data=viz_df, x="age_group", hue="survived")

plt.xticks(rotation=45)
plt.title("Survival Count by Age Group (5-Year Bins)")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------------------------------------------
# 3. DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------------

# convert df to numpy
X_df = df.drop(columns=["survived", "2nd_class", "family_size"])
X = X_df.values
y = df["survived"].values


def train_test_split_numpy(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_index = int(len(X) * (1 - test_size))

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    return (X[train_indices], X[test_indices], y[train_indices], y[test_indices])


X_train, X_test, y_train, y_test = train_test_split_numpy(X, y)

# ---------------------------------------------------------------
# 5. STANDARD SCALER (NUMPY IMPLEMENTATION)
# ---------------------------------------------------------------

numeric_indices = [1, 2]  # age, fare


def standardize_fit(X, numeric_indices):
    means = np.mean(X[:, numeric_indices], axis=0)
    stds = np.std(X[:, numeric_indices], axis=0)

    print(means, stds)

    return means, stds


def standardize_transform(X, numeric_indices, means, stds):
    X_copy = X.copy().astype(float)

    X_copy[:, numeric_indices] = (X_copy[:, numeric_indices] - means) / stds

    return X_copy


means, stds = standardize_fit(X_train, numeric_indices)

X_train_scaled = standardize_transform(X_train, numeric_indices, means, stds)

X_test_scaled = standardize_transform(X_test, numeric_indices, means, stds)

# ---------------------------------------------------------------
# 6. LOGISTIC REGRESSION
# ---------------------------------------------------------------


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=5000, C=1.0, fit_intercept=True):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.fit_intercept = fit_intercept

        self.weights = None
        self.bias = None

    def sigmoid(self, z):

        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def compute_loss(self, y, y_hat):

        epsilon = 1e-15

        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        data_loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        # L2 regularization term
        reg_loss = (1 / (2 * self.C * len(y))) * np.sum(self.weights**2)

        return data_loss + reg_loss

    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Better initialization
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Linear combination
            linear_model = np.dot(X, self.weights)

            if self.fit_intercept:
                linear_model += self.bias

            # Probabilities
            y_hat = self.sigmoid(linear_model)

            # Errors
            errors = y_hat - y

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, errors)

            # Add L2 regularization gradient
            dw += (1 / (self.C * n_samples)) * self.weights

            db = (1 / n_samples) * np.sum(errors)

            # Parameter updates
            self.weights -= self.learning_rate * dw

            if self.fit_intercept:
                self.bias -= self.learning_rate * db

            # Print progress
            if epoch % 50 == 0:
                loss = self.compute_loss(y, y_hat)

                print(f"Epoch {epoch} | Loss: {loss:.6f}")

    def predict_proba(self, X):

        linear_model = np.dot(X, self.weights)

        if self.fit_intercept:
            linear_model += self.bias

        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):

        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)


# ---------------------------------------------------------------
# 7. MODEL TRAINING
# ---------------------------------------------------------------

model = LogisticRegressionScratch(learning_rate=0.5, epochs=500, C=10.0)

model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------
# 8. PREDICTIONS
# ---------------------------------------------------------------

y_prob = model.predict_proba(X_test_scaled)
y_pred = model.predict(X_test_scaled)

# ---------------------------------------------------------------
# 9. EVALUATION METRICS (NUMPY IMPLEMENTATION)
# ---------------------------------------------------------------


def confusion_matrix_numpy(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[tn, fp], [fn, tp]])


def accuracy_score_numpy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score_numpy(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    return tp / (tp + fp + 1e-15)


def recall_score_numpy(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp / (tp + fn + 1e-15)


def f1_score_numpy(y_true, y_pred):

    precision = precision_score_numpy(y_true, y_pred)
    recall = recall_score_numpy(y_true, y_pred)

    return 2 * (precision * recall) / (precision + recall + 1e-15)


def roc_curve_numpy(y_true, y_scores):

    thresholds = np.sort(np.unique(y_scores))[::-1]

    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        tpr = tp / (tp + fn + 1e-15)
        fpr = fp / (fp + tn + 1e-15)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def roc_auc_score_numpy(fpr, tpr):
    return np.trapz(tpr, fpr)


# ---------------------------------------------------------------
# 10. MODEL EVALUATION
# ---------------------------------------------------------------

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix_numpy(y_test, y_pred)

print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

accuracy = accuracy_score_numpy(y_test, y_pred)
precision = precision_score_numpy(y_test, y_pred)
recall = recall_score_numpy(y_test, y_pred)
f1 = f1_score_numpy(y_test, y_pred)

fpr, tpr, thresholds = roc_curve_numpy(y_test, y_prob)
roc_auc = roc_auc_score_numpy(fpr, tpr)

print("\n===== METRICS =====")
print(f"Accuracy Score : {accuracy:.2f}")
print(f"Precision Score: {precision:.2f}")
print(f"Recall Score   : {recall:.2f}")
print(f"F1 Score       : {f1:.2f}")
print(f"ROC AUC Score  : {roc_auc:.2f}")

# ---------------------------------------------------------------
# 11. ROC CURVE
# ---------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

plt.plot([0, 1], [0, 1], linestyle="--", color="red")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ---------------------------------------------------------------
# 12. FEATURE IMPORTANCE
# ---------------------------------------------------------------

coefficients = pd.DataFrame({"Feature": X_df.columns, "Coefficient": model.weights})

coefficients = coefficients.sort_values(by="Coefficient", ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(coefficients)

plt.figure(figsize=(10, 6))

sns.barplot(x="Coefficient", y="Feature", data=coefficients)

plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.show()
