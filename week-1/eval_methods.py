import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)


class ModelEvaluation:

    def evaluate(self, y_test, y_pred, y_prob):

        print("\n===== CONFUSION MATRIX =====")

        cm = confusion_matrix(y_test, y_pred)

        print(cm)

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues'
        )

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.show()

        print("\n===== CLASSIFICATION REPORT =====")

        print(classification_report(y_test, y_pred))

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"Accuracy Score : {accuracy:.2f}")
        print(f"Precision Score: {precision:.2f}")
        print(f"Recall Score   : {recall:.2f}")
        print(f"F1 Score       : {f1:.2f}")
        print(f"ROC AUC Score  : {roc_auc:.2f}")

        return roc_auc

    def roc_curve_plot(self, y_test, y_prob, roc_auc):

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        plt.figure(figsize=(7, 5))

        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle='--',
            color='red'
        )

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.title('ROC Curve')

        plt.legend()

        plt.show()

    def feature_importance(self, X, model):

        coefficients = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_[0]
        }).sort_values(
            by='Coefficient',
            ascending=False
        )

        print("\n===== FEATURE IMPORTANCE =====")

        print(coefficients)

        plt.figure(figsize=(10, 6))

        sns.barplot(
            x='Coefficient',
            y='Feature',
            data=coefficients
        )

        plt.title(
            'Feature Importance (Logistic Regression Coefficients)'
        )

        plt.show()