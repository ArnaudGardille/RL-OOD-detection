"""Sliding-window Kolmogorov-Smirnov baseline detector.

Compares the last ``window`` nonconformity scores to an in-distribution reference via a
two-sample KS statistic; alarms when it exceeds ``h``. Like CUSUM, KS is a *running*
detector, so ``h`` is calibrated at the **stream** level — the ``(1 - delta)`` quantile of
the per-stream maximum KS statistic over a batch of nominal runs — to avoid the
multiple-testing false-alarm inflation. The reference set and the null-calibration runs are
disjoint (different nominal streams), since scoring a window against the set it was drawn
from underestimates the statistic.
"""

from collections import deque

import numpy as np
from scipy.stats import ks_2samp

from .base import Detector

_MAX_REF = 4000  # cap the reference set for KS speed


class KSWindowDetector(Detector):
    def __init__(self, window: int = 100, delta: float = 0.01, eval_every: int = 5):
        self.window = int(window)
        self.delta = float(delta)
        self.eval_every = max(1, int(eval_every))
        self.cal = None
        self.h = None
        self.reset()

    def reset(self) -> None:
        self.buf = deque(maxlen=self.window)
        self._stat = 0.0
        self._t = 0

    def _stream_max(self, run) -> float:
        buf = deque(maxlen=self.window)
        mx = 0.0
        for t, x in enumerate(run):
            buf.append(float(x))
            if len(buf) >= self.window and t % self.eval_every == 0:
                stat = ks_2samp(np.fromiter(buf, dtype=float), self.cal).statistic
                if stat > mx:
                    mx = stat
        return mx

    def calibrate(self, streams, rng=0) -> None:
        a = np.asarray(streams, dtype=float)
        runs = list(a) if a.ndim == 2 else [a]
        half = max(1, len(runs) // 2)
        ref = np.concatenate(runs[:half])
        if ref.size > _MAX_REF:  # subsample reference for speed
            ref = np.random.default_rng(rng).choice(ref, _MAX_REF, replace=False)
        self.cal = ref
        null_runs = runs[half:] if len(runs) > 1 else runs
        maxes = np.array([self._stream_max(run) for run in null_runs])
        self.h = float(np.quantile(maxes, 1.0 - self.delta))

    def update(self, alpha_t: float) -> float:
        self.buf.append(float(alpha_t))
        self._t += 1
        if len(self.buf) >= self.window and self._t % self.eval_every == 0:
            self._stat = float(ks_2samp(np.fromiter(self.buf, dtype=float), self.cal).statistic)
        return self._stat

    def alarm(self) -> bool:
        return self.h is not None and self._stat > self.h
