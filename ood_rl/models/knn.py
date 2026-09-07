"""KNN dynamics model — the reference world-model from the original repo.

Note: ``KNeighborsRegressor`` is natively multi-output, so we drop the redundant
``MultiOutputRegressor`` wrapper the legacy code used (old ``rl_ood.py``: one KNN per
output dimension, slower for no benefit).
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from .base import WorldModel


class KNNDynamics(WorldModel):
    def __init__(self, n_neighbors: int = 5, **knn_kwargs):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, **knn_kwargs)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X))
