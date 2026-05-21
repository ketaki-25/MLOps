import numpy as np

class LogisticRegressionScratch:

    def __init__(
        self,
        learning_rate=0.01,
        epochs=5000,
        C=1.0,
        fit_intercept=True
    ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.fit_intercept = fit_intercept

        self.weights = None
        self.bias = None

    def sigmoid(self, z):

        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

    def compute_loss(self, y, y_hat):

        epsilon = 1e-15

        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        data_loss = -np.mean(
            y * np.log(y_hat) +
            (1 - y) * np.log(1 - y_hat)
        )

        # L2 regularization term
        reg_loss = (1 / (2 * self.C * len(y))) * np.sum(self.weights ** 2)

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