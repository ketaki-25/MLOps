import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------

df = pd.read_csv("data/titanic.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ---------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------

# Create a child feature
df["child"] = (df["age"] < 16).astype(int)

# ---------------------------------------------------------------
# 3. DEFINE FEATURES & TARGET
# ---------------------------------------------------------------

X = df.drop(columns=["survived", "2nd_class"])
y = df["survived"]

# ---------------------------------------------------------------
# 4. TRAIN TEST SPLIT
# ---------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

from sklearn.preprocessing import OneHotEncoder

# Separate feature types
numeric_features = ["age", "fare", "family_size"]

categorical_features = ["sex", "1st_class", "3rd_class", "child"]

# Numeric pipeline
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Categorical pipeline
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first")),
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ---------------------------------------------------------------
# 6. CREATE PIPELINE
# ---------------------------------------------------------------

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced"),
        ),
    ]
)

# ---------------------------------------------------------------
# 7. GRID SEARCH
# ---------------------------------------------------------------

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__penalty": ["l2"],
    "classifier__solver": ["lbfgs"],
}


grid_search = GridSearchCV(
    estimator=pipeline, param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1
)

# Train models
grid_search.fit(X_train, y_train)

# ---------------------------------------------------------------
# 8. BEST MODEL
# ---------------------------------------------------------------

best_model = grid_search.best_estimator_

print("\n===== BEST PARAMETERS =====")
print(grid_search.best_params_)

print("\n===== BEST CROSS VALIDATION F1 SCORE =====")
print(grid_search.best_score_)

# ---------------------------------------------------------------
# 9. CROSS VALIDATION SCORES
# ---------------------------------------------------------------

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy")

print("\n===== CROSS VALIDATION SCORES =====")
print(cv_scores)

print("\nMean CV Accuracy:", cv_scores.mean())

# ---------------------------------------------------------------
# 10. PREDICTIONS
# ---------------------------------------------------------------

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------
# 11. CONFUSION MATRIX
# ---------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n===== CONFUSION MATRIX =====")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ---------------------------------------------------------------
# 12. CLASSIFICATION REPORT
# ---------------------------------------------------------------

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------
# 13. METRICS
# ---------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\nAccuracy Score: {accuracy:.2f}")
print(f"Precision Score: {precision:.2f}")
print(f"Recall Score: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")

# ---------------------------------------------------------------
# 14. ROC CURVE
# ---------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

plt.plot([0, 1], [0, 1], linestyle="--", color="red")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# ---------------------------------------------------------------
# 15. FEATURE IMPORTANCE
# ---------------------------------------------------------------

feature_names = X.columns

coefficients = best_model.named_steps["classifier"].coef_[0]

feature_importance = pd.DataFrame(
    {"Feature": feature_names, "Coefficient": coefficients}
)

feature_importance = feature_importance.sort_values(by="Coefficient", ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(feature_importance)

plt.figure(figsize=(10, 6))

sns.barplot(x="Coefficient", y="Feature", data=feature_importance)

plt.title("Feature Importance")

plt.show()
