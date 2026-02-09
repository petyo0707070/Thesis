
import numpy as np

class GRNN:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data (GRNN uses a memory-based approach).
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def _kernel(self, X):
        """
        Gaussian radial basis function.
        """
        diff = self.X_train - X
        dist_sq = np.sum(diff ** 2, axis=1)
        return np.exp(-dist_sq / (2 * self.sigma ** 2))

    def predict(self, X):
        """
        Predict outputs for new samples.
        """
        X = np.asarray(X)
        preds = []

        for x in X:
            w = self._kernel(x)
            numerator = np.sum(w * self.y_train)
            denominator = np.sum(w)

            # Avoid division by zero
            preds.append(numerator / (denominator + 1e-12))

        return np.array(preds)
