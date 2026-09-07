"""Detector interface: consume a stream of per-step nonconformity scores."""

from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    """Online change/OOD detector over a stream of nonconformity scores ``alpha_t``."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the running statistic to its initial state."""

    @abstractmethod
    def calibrate(self, alpha_cal: np.ndarray) -> None:
        """Provide in-distribution nonconformity scores for calibration."""

    @abstractmethod
    def update(self, alpha_t: float) -> float:
        """Ingest one score, return the current detection statistic."""

    @abstractmethod
    def alarm(self) -> bool:
        """Whether the statistic has crossed the alarm threshold."""

    def run(self, alphas: np.ndarray):
        """Stream a whole sequence. Returns ``(scores, alarm_at)``.

        ``alarm_at`` is the index of the first alarm, or ``-1`` if none fired.
        """
        self.reset()
        alphas = np.asarray(alphas)
        scores = np.empty(alphas.shape[0], dtype=float)
        alarm_at = -1
        for i, a in enumerate(alphas):
            scores[i] = self.update(float(a))
            if alarm_at < 0 and self.alarm():
                alarm_at = i
        return scores, alarm_at
