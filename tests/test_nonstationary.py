"""Intra-trajectory change-point: the dynamics switch at tau, and the (online) detector
fires *after* tau with a measurable delay — the metric Phase 0 could not produce."""

import numpy as np
import jax

from ood_rl import make_env, Phase0Config
from ood_rl.envs.params import build_params
from ood_rl.experiment.rollout import fit_reference_detector, collect_nonstationary, detect


def _run(env, pb, pa, t_change, model, detector, n_seeds, T):
    alarms, delays = 0, []
    for s in range(n_seeds):
        k = jax.random.fold_in(jax.random.PRNGKey(7), s)
        X, y, _r, ci = collect_nonstationary(env, pb, pa, t_change, k, T, memory_size=10)
        detector.rng = np.random.default_rng(100 + s)
        _, at = detect(model, detector, X, y)
        if at >= 0:
            alarms += 1
            delays.append(at - ci)
    return alarms / n_seeds, (float(np.mean(delays)) if delays else float("nan"))


def test_changepoint_detection_delay():
    cfg = Phase0Config("CartPole-v1", n_train=6_000, n_cal=3_000, delta=0.01, stride=1)
    env, dp, _ = make_env(cfg.env_name)
    model, detector = fit_reference_detector(env, dp, jax.random.PRNGKey(0), cfg, rng=0)

    T, tau = 500, 200
    ood = build_params(cfg.env_name, dp, gravity=9.8 * 10.0)

    rate, mean_delay = _run(env, dp, ood, tau, model, detector, n_seeds=12, T=T)
    assert rate >= 0.8, f"change-point not detected often enough: {rate:.2f}"
    assert 0 <= mean_delay < (T - tau), f"implausible mean delay: {mean_delay}"

    # No-shift control: same params before/after -> few alarms (FPR controlled).
    ctrl_rate, _ = _run(env, dp, dp, tau, model, detector, n_seeds=12, T=T)
    assert ctrl_rate <= 0.25, f"too many alarms with no shift: {ctrl_rate:.2f}"
