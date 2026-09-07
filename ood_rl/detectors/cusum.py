"""CUSUM baseline detector on the nonconformity stream.

Classic one-sided cumulative-sum change detector: ``S_t = max(0, S_{t-1} + (alpha_t - k))``,
alarm when ``S_t > h``. Calibrated on the *same* nominal scores as the other detectors.

Because CUSUM is evaluated at every step, its stream-level false-alarm rate is subject to
multiple testing — so ``h`` is calibrated at the **stream** level: the ``(1 - delta)``
quantile of the per-stream supremum of ``S_t`` over a batch of nominal runs. (The conformal
martingale gets this sup-level control for free via Ville's inequality.)
"""

import numpy as np

from .base import Detector


class CUSUMDetector(Detector):
    def __init__(self, k_sigma: float = 0.5, delta: float = 0.01, seg_len: int = 500):
        self.k_sigma = float(k_sigma)
        self.delta = float(delta)
        self.seg_len = int(seg_len)  # used only when calibrating from a flat 1-D stream
        self.k = None
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
        for x in run:
            S = max(0.0, S + (x - self.k))
            if S > m:
                m = S
        return m

    def calibrate(self, streams) -> None:
        a = np.asarray(streams, dtype=float)
        flat = a.ravel()
        self.k = float(flat.mean() + self.k_sigma * flat.std())
        sups = np.array([self._stream_sup(run) for run in self._runs(a)])
        self.h = float(np.quantile(sups, 1.0 - self.delta))
        if self.h <= 0.0:
            self.h = float(sups.max() + 1e-6)

    def update(self, alpha_t: float) -> float:
        self.S = max(0.0, self.S + (float(alpha_t) - self.k))
        return self.S

    def alarm(self) -> bool:
        return self.h is not None and self.S > self.h
