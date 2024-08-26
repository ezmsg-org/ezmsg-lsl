from dataclasses import dataclass
import typing

import numpy as np
import numpy.typing as npt


@dataclass
class LinearRegressionState:
    weights: npt.NDArray
    bias: float
    X0: npt.NDArray
    y0: float


class LinearRegressionSGD:
    def __init__(
        self,
        state: typing.Optional[LinearRegressionState] = None,
        learning_rate: float = 0.01,
    ):
        self.state = state
        self.learning_rate = learning_rate

    def _update(self, x, y, y_pred):
        db = y_pred - y
        dw = db * (x - self.state.X0)
        self.state.bias -= self.learning_rate * db
        self.state.weights -= self.learning_rate * dw

    def partial_fit(self, X, y):
        if self.state is None:
            self.state = LinearRegressionState(
                weights=np.ones_like(X[:1], dtype=float),
                bias=0,
                X0=X[:1],
                y0=y[0],
            )

        for ix in range(X.shape[0]):
            sample = X[ix:ix+1]
            y_pred = self.predict(sample)
            self._update(sample, y[ix], y_pred)

    def predict(self, X):
        return np.dot(X - self.state.X0, self.state.weights)[:, 0] + self.state.bias + self.state.y0
