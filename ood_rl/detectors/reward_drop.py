"""Reward-drop baseline: detect OOD by watching a trained policy's reward fall.

The "naive" performance-based detector that dynamics-based detection should beat. It is a
one-sided CUSUM on the *negative* reward deviation from the calibrated nominal mean:
``S_t = max(0, S_{t-1} + (mu - r_t) - slack)``, alarm when ``S_t > h``. Like the other
running detectors, ``h`` is calibrated at the stream level (sup over nominal runs).

It consumes the policy's per-step reward stream, so it only sees a shift if the shift
actually degrades the policy — and it needs an informative (dense) reward. For sparse /
constant per-step rewards (e.g. CartPole gives +1 every step regardless of dynamics) the
signal must instead be episode return; that is left for a later phase.
"""

import numpy as np

from .base import Detector


class RewardDropDetector(Detector):
    def __init__(self, delta: float = 0.01, k_sigma: float = 0.5, seg_len: int = 500):
        self.delta = float(delta)
        self.k_sigma = float(k_sigma)
        self.seg_len = int(seg_len)
        self.mu = None
        self.slack = None
        self.h = None
        self.reset()

    def reset(self) -> None:
        self.S = 0.0

    def _runs(self, a):
        if a.ndim == 2:
            return list(a)
        return np.array_split(a, max(1, a.size // self.seg_len))

    def _stream_sup(self, run) -> float:
        S = 0.0
        m = 0.0
        for r in run:
            S = max(0.0, S + ((self.mu - r) - self.slack))
            if S > m:
                m = S
        return m

    def calibrate(self, reward_streams) -> None:
        a = np.asarray(reward_streams, dtype=float)
        flat = a.ravel()
        self.mu = float(flat.mean())
        self.slack = self.k_sigma * float(flat.std())
        sups = np.array([self._stream_sup(run) for run in self._runs(a)])
        self.h = float(np.quantile(sups, 1.0 - self.delta))
        if self.h <= 0.0:
            self.h = float(sups.max() + 1e-6)

    def update(self, r_t: float) -> float:
        self.S = max(0.0, self.S + ((self.mu - float(r_t)) - self.slack))
        return self.S

    def alarm(self) -> bool:
        return self.h is not None and self.S > self.h
