"""Baselines (CUSUM, KS) must control the stream-level false-alarm rate on the nominal env
and fire on a strong OOD shift — the same bar as the conformal detector, so the benchmark
comparison is fair."""

import numpy as np
import jax

from ood_rl import make_env, Phase0Config, CUSUMDetector, KSWindowDetector
from ood_rl.envs.params import build_params
from ood_rl.experiment import sweep, metrics


def _alarm_rate(detector, env, params, key, cfg, n_streams, model):
    X, y, _ = sweep.sweep_streams(env, [{"params": params}] * n_streams, key, cfg)
    alphas = sweep.alpha_streams(model, X, y, stride=cfg.stride)
    return metrics.detection_rate([at for at, _ in sweep.run_detector_on_streams(detector, alphas)])


def test_baselines_control_fpr_and_detect():
    cfg = Phase0Config("CartPole-v1", n_train=5_000, n_test=400, delta=0.05, stride=1)
    env, dp, _ = make_env(cfg.env_name)
    k_fit, k_nom, k_ood = jax.random.split(jax.random.PRNGKey(0), 3)

    model, dets = sweep.fit_detectors(
        env, dp, k_fit, cfg,
        {
            "cusum": lambda: CUSUMDetector(delta=cfg.delta, seg_len=cfg.n_test),
            "ks": lambda: KSWindowDetector(window=80, delta=cfg.delta, eval_every=10),
        },
        n_cal_streams=60,
    )

    ood = build_params(cfg.env_name, dp, gravity=9.8 * 10.0)
    for name, det in dets.items():
        fpr = _alarm_rate(det, env, dp, k_nom, cfg, 25, model)
        power = _alarm_rate(det, env, ood, k_ood, cfg, 25, model)
        assert fpr <= 0.25, f"{name}: FPR not controlled ({fpr:.2f})"
        assert power >= 0.8, f"{name}: insufficient sensitivity to strong OOD ({power:.2f})"
