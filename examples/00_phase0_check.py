"""Phase 0 sanity check.

Demonstrates the two properties that define Phase 0 done:
  1. the corrected conformal-martingale detector *controls the false-alarm rate* on the
     nominal env (the legacy detector did not), and
  2. it *fires* on out-of-distribution dynamics, faster as the shift grows.

Run:  python examples/00_phase0_check.py
"""

import numpy as np
import jax

from ood_rl.envs.registry import make_env
from ood_rl.envs.params import build_params, DEFAULTS
from ood_rl.experiment.config import Phase0Config
from ood_rl.experiment.rollout import collect_transitions, fit_reference_detector, detect


def _alarm_stats(env, params, model, detector, base_key, cfg, n_seeds):
    alarms, delays = 0, []
    for s in range(n_seeds):
        k = jax.random.fold_in(base_key, s)
        X, y = collect_transitions(env, params, k, cfg.n_test, cfg.memory_size, cfg.stride)
        detector.rng = np.random.default_rng(10_000 + s)  # reproducible smoothing
        _, at = detect(model, detector, X, y)
        if at >= 0:
            alarms += 1
            delays.append(at)
    rate = alarms / n_seeds
    delay = float(np.mean(delays)) if delays else float("nan")
    return rate, delay


def evaluate_env(env_name, knob, scales, n_seeds=10, stride=1):
    cfg = Phase0Config(env_name=env_name, n_train=8_000, n_cal=3_000, n_test=800,
                       delta=0.01, stride=stride)
    env, default_params, _ = make_env(env_name)
    fit_key, eval_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
    model, detector = fit_reference_detector(env, default_params, fit_key, cfg, rng=0)

    print(f"\n=== {env_name} (delta={cfg.delta}, threshold logM>={np.log(1/cfg.delta):.2f}, "
          f"stride={cfg.stride}) ===")
    fpr, _ = _alarm_stats(env, default_params, model, detector,
                          jax.random.fold_in(eval_key, 0), cfg, n_seeds)
    print(f"  nominal              : false-alarm rate = {fpr:.2f}   (target <= {cfg.delta})")
    print(f"  OOD knob '{knob}' (default {DEFAULTS[env_name][knob]}):")
    for sc in scales:
        params = build_params(env_name, default_params, **{knob: DEFAULTS[env_name][knob] * sc})
        rate, delay = _alarm_stats(env, params, model, detector,
                                   jax.random.fold_in(eval_key, int(sc * 1000) + 1), cfg, n_seeds)
        delay_str = f"{delay:5.0f} steps" if not np.isnan(delay) else "    n/a   "
        print(f"    x{sc:<5} : detection rate = {rate:.2f},  mean delay = {delay_str}")


if __name__ == "__main__":
    # CartPole terminates early, which periodically resets temporal correlation, so
    # stride=1 already yields calibrated p-values. Pendulum never terminates and its
    # transitions are strongly autocorrelated (exchangeability violation) — stride=5
    # decorrelates them and restores FPR control. This is the documented caveat/mitigation.
    evaluate_env("CartPole-v1", "gravity", [0.5, 2.0, 5.0, 10.0], stride=1)
    evaluate_env("Pendulum-v1", "g", [0.5, 2.0, 5.0, 10.0], stride=5)
