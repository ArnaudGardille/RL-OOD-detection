"""Online conformal test-martingale OOD detector (corrected).

This replaces the legacy ``MartingaleOODDetector`` (old ``rl_ood.py``), which had two
coupled defects:

1. p-values were parametric (``2*norm.cdf(-|err|)``) on *unnormalized* residuals, with
   no false-alarm-rate control.
2. the martingale was computed as one batch product over hundreds of p-values < 1, which
   underflows to 0, then numerically integrated — unstable and not online.

Here we use distribution-free **rank-based conformal p-values** against an
in-distribution calibration set, accumulate a **power test-martingale in log-space**
(online), and alarm via **Ville's inequality**, which gives anytime-valid control of the
false-alarm probability at level ``delta``:

    P( sup_t  M_t  >=  1/delta )  <=  delta      (for a nonnegative martingale, M_0 = 1)

so the alarm rule ``log M_t >= log(1/delta)`` controls the FPR — the property the legacy
ad-hoc ``score + 10*std`` threshold lacked.

Caveat (documented honestly): consecutive transitions in a rollout are temporally
correlated, violating the exchangeability assumption. The empirical FPR can therefore sit
slightly above ``delta``. Mitigations: subsample transitions (``stride`` in the rollout)
and compare against the CUSUM/KS baselines added in Phase 1.
"""

import numpy as np


class ConformalMartingaleDetector:
    """Power martingale over smoothed conformal p-values, with a Ville threshold.

    Args:
        epsilon: power-martingale exponent in (0, 1). The per-step betting factor is
            ``epsilon * p**(epsilon - 1)``, whose expectation under uniform p is 1.
        delta: target false-alarm level; alarm when ``M_t >= 1/delta``.
        rng: seed / Generator for the smoothing randomization of conformal p-values.
    """

    def __init__(self, epsilon: float = 0.92, delta: float = 0.01, rng=None):
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be in (0, 1)")
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.log_threshold = np.log(1.0 / self.delta)
        self.rng = np.random.default_rng(rng)
        self._cal = None
        self.n_cal = 0
        self.reset()

    def reset(self) -> None:
        self.log_M = 0.0  # M_0 = 1

    def calibrate(self, alpha_cal: np.ndarray) -> None:
        # Accepts a flat array or a [N, T] batch of nominal streams (flattened to the
        # exchangeable reference set). The alarm threshold is the analytic Ville bound, so
        # — unlike the running baselines — no stream-level sup-calibration is needed.
        self._cal = np.sort(np.asarray(alpha_cal, dtype=float).ravel())
        self.n_cal = self._cal.size
        if self.n_cal == 0:
            raise ValueError("calibration set is empty")

    def _p_value(self, alpha_t: float) -> float:
        """Smoothed conformal p-value for a right-tailed (larger = more anomalous) score.

        p = ( #{cal > a} + U * (#{cal == a} + 1) ) / (n_cal + 1),  U ~ Uniform(0, 1),
        which is exactly uniform on (0, 1) under exchangeability.
        """
        cal = self._cal
        n_ge = self.n_cal - np.searchsorted(cal, alpha_t, side="left")   # #{cal >= a}
        n_gt = self.n_cal - np.searchsorted(cal, alpha_t, side="right")  # #{cal > a}
        ties = n_ge - n_gt                                               # #{cal == a}
        u = self.rng.random()
        p = (n_gt + u * (ties + 1)) / (self.n_cal + 1)
        # Floor for numerical safety (min attainable p ~ 1/(n_cal+1)).
        return max(p, 1.0 / (self.n_cal + 1))

    def update(self, alpha_t: float) -> float:
        if self._cal is None:
            raise RuntimeError("call calibrate() before update()")
        p = self._p_value(alpha_t)
        # log betting factor of the power martingale: log(eps) + (eps - 1) * log(p)
        self.log_M += np.log(self.epsilon) + (self.epsilon - 1.0) * np.log(p)
        return self.log_M

    def alarm(self) -> bool:
        return self.log_M >= self.log_threshold

    # Convenience streaming runner (mirrors Detector.run).
    def run(self, alphas: np.ndarray):
        self.reset()
        alphas = np.asarray(alphas)
        scores = np.empty(alphas.shape[0], dtype=float)
        alarm_at = -1
        for i, a in enumerate(alphas):
            scores[i] = self.update(float(a))
            if alarm_at < 0 and self.alarm():
                alarm_at = i
        return scores, alarm_at
