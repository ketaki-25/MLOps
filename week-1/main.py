from numpy_logistic_regression import LogisticRegressionScratch
from visualisations import EDA
import pandas as pd
from preprocessing import Preprocessor
from sklearn.linear_model import LogisticRegression

def eda_graphs(df):
    eda = EDA(df)
    eda.run_all()

def numpyModel(df):

    preprocessor = Preprocessor()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        x_train_standardized,
        x_test_standardized
    ) = preprocessor.process(df)

    model = LogisticRegression(
        max_iter=200,
        class_weight='balanced'
    )
    model.fit(x_train_standardized, y_train)

    y_pred = model.predict(x_test_standardized)
    y_prob = model.predict_proba(x_test_standardized)[:, 1]


def skModel(df):
    ...


if __name__ == '__main__':
    df = pd.read_csv('data/titanic.csv')
    eda_graphs(df)

    numpyModel(df)
    skModel(df)