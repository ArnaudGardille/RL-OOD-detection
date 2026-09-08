"""World-model interface: predict next-state delta, expose a per-step nonconformity.

The nonconformity score is the building block consumed by every detector. We use the
L2 norm of the *standardized* prediction residual (standardized per output dimension by
the training-residual std). This is cleaner than the legacy approach, which flattened
every ``(step, dim)`` residual into a separate parametric p-value (old ``rl_ood.py``).
"""

from abc import ABC, abstractmethod

import numpy as np


class WorldModel(ABC):
    """Predicts ``y = s' - s`` from a flattened (obs, action) history window ``X``."""

    _resid_std: np.ndarray

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the underlying regressor (subclass-specific)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-state deltas, shape ``(n, obs_dim)``."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WorldModel":
        self._fit(X, y)
        resid = self.predict(X) - y
        # Floor avoids division by zero on perfectly-predicted dimensions.
        self._resid_std = np.maximum(resid.std(axis=0), 1e-8)
        return self

    def residual(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-step scalar nonconformity score, shape ``(n,)`` (larger = more anomalous)."""
        resid = self.predict(X) - y
        return np.linalg.norm(resid / self._resid_std, axis=1)
