import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve, f1_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------
df = pd.read_csv('data/titanic.csv')

print(df.describe())
print(df.isnull().sum())


# ---------------------------------------------------------------
# 2. DATA EXPLORATION
# ---------------------------------------------------------------
numeric_features = ["age", "family_size", "fare"]
binary_features = ["sex", "1st_class", "2nd_class", "3rd_class"]

for features in df.columns:
    if features in binary_features:
        sns.barplot(x=features, y='survived', data=df)
        plt.title('Survival Rate by {features}'.format(features=features))
        plt.show()

# Age distribution
plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

#Age group by survivors
viz_df = df.copy()
bins = np.arange(0, viz_df['age'].max() + 5, 5)
viz_df['age_group'] = pd.cut(viz_df['age'], bins=bins)
plt.figure(figsize=(12,6))

sns.countplot(
    data=viz_df,
    x='age_group',
    hue='survived'
)
plt.xticks(rotation=45)
plt.title('Survival Count by Age Group (5-Year Bins)')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()


# Correlation heatmap for numerical columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



# ---------------------------------------------------------------
# 3. DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------------

# Define Features (X) and Target (Y)
X = df.drop(columns=['survived', '2nd_class', 'family_size'])
y = df['survived']

# Split the data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ["age", "fare"]
binary_features = ["sex", "1st_class", "3rd_class"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
    ],
    remainder="passthrough"
)

x_train_standardized = preprocessor.fit_transform(X_train)
x_test_standardized = preprocessor.transform(X_test)




# ---------------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------------
model = LogisticRegression(
    max_iter=200,
    class_weight='balanced'
)
model.fit(x_train_standardized, y_train)

# ---------------------------------------------------------------
# 5. PREDICTIONS
# ---------------------------------------------------------------
y_pred = model.predict(x_test_standardized)
y_prob = model.predict_proba(x_test_standardized)[:, 1]


# ---------------------------------------------------------------
# 6. MODEL EVALUATION
# ---------------------------------------------------------------
print("\n===== CONFUSION MATRIX =====")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.2f}")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision Score: {precision:.2f}")

# Recall
recall = recall_score(y_test, y_pred)
print(f"Recall Score: {recall:.2f}")

# F1 Score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score: {roc_auc:.2f}")

# ---------------------------------------------------------------
# 7. ROC CURVE
# ---------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# ---------------------------------------------------------------
# 8. FEATURE IMPORTANCE (Logistic Regression Coefficients)
# ---------------------------------------------------------------

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

coefficients = coefficients.sort_values(by='Coefficient', ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(coefficients)

plt.figure(figsize=(10,6))
sns.barplot(
    x='Coefficient',
    y='Feature',
    data=coefficients
)

plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.show()
