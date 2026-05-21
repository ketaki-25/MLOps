from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


class Preprocessor:

    def __init__(self):

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ["fare", "age"])
            ],
            remainder="passthrough"
        )

    def process(self, df):

        X = df.drop(columns=['survived', '2nd_class', 'family_size'])

        y = df['survived']

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=True,
            stratify=y,
            random_state=42
        )

        x_train_standardized = self.preprocessor.fit_transform(X_train)
        x_test_standardized = self.preprocessor.transform(X_test)

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            x_train_standardized,
            x_test_standardized
        )