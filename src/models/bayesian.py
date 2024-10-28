import torch
import numpy as np
from sklearn.linear_model import LinearRegression

class BayesianNetworkComponent:
    """Bayesian Network component using linear regression for continuous data."""

    def __init__(self, n_variables: int, structure: Optional[List[Tuple[str, str]]] = None):
        self.n_variables = n_variables
        self.models = [LinearRegression() for _ in range(n_variables)]
        self.fitted = False  # Flag to track if the models are fitted

    def fit(self, data: np.ndarray):
        """Fit the Bayesian Network models."""
        if data.shape[1] != self.n_variables:
            raise ValueError(f"Data dimension mismatch. Expected {self.n_variables}, got {data.shape[1]}.")
        for i in range(self.n_variables):
            X = data[:, :i] if i > 0 else np.ones((data.shape[0], 1)) #Use ones for the first variable
            y = data[:, i]
            self.models[i].fit(X, y)
        self.fitted = True

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.fitted:
            raise RuntimeError("Bayesian Network models must be fitted before prediction.")
        predictions = np.zeros_like(input_data)
        for i in range(self.n_variables):
            X = input_data[:, :i] if i > 0 else np.ones((input_data.shape[0], 1))
            predictions[:, i] = self.models[i].predict(X)
        return predictions