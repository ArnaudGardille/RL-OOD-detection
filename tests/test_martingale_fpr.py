"""The Phase 0 proof: the corrected detector controls the false-alarm rate on the nominal
env (which the legacy detector did not) and stays sensitive to a strong OOD shift.
"""

import numpy as np
import jax

from ood_rl import make_env, Phase0Config
from ood_rl.envs.params import build_params
from ood_rl.experiment.rollout import collect_transitions, fit_reference_detector, detect


def _alarm_fraction(env, params, model, detector, base_key, n_seeds, cfg):
    alarms = 0
    for s in range(n_seeds):
        k = jax.random.fold_in(base_key, s)
        X, y = collect_transitions(env, params, k, cfg.n_test, cfg.memory_size, cfg.stride)
        detector.rng = np.random.default_rng(7_000 + s)
        _, at = detect(model, detector, X, y)
        alarms += int(at >= 0)
    return alarms / n_seeds


def test_fpr_control_and_sensitivity():
    cfg = Phase0Config(env_name="CartPole-v1", n_train=6_000, n_cal=2_500, n_test=600, delta=0.05)
    env, default_params, _ = make_env(cfg.env_name)
    model, detector = fit_reference_detector(env, default_params, jax.random.PRNGKey(0), cfg, rng=0)

    n_seeds = 20
    fpr = _alarm_fraction(env, default_params, model, detector, jax.random.PRNGKey(100), n_seeds, cfg)
    # Target is delta=0.05; allow slack for finite samples + exchangeability violation.
    # A *broken* calibration (the legacy failure mode) would land near ~1.0 here.
    assert fpr <= 0.25, f"false-alarm rate not controlled: {fpr:.2f}"

    ood = build_params(cfg.env_name, default_params, gravity=9.8 * 10.0)
    power = _alarm_fraction(env, ood, model, detector, jax.random.PRNGKey(200), n_seeds, cfg)
    assert power >= 0.8, f"detector insufficiently sensitive to strong OOD: {power:.2f}"


def test_fpr_control_pendulum_needs_stride():
    """Pendulum never terminates -> strongly autocorrelated transitions break
    exchangeability. stride=1 inflates the FPR; stride>=5 restores control. This guards
    the documented mitigation."""
    env, default_params, _ = make_env("Pendulum-v1")
    n_seeds = 15

    cfg1 = Phase0Config(env_name="Pendulum-v1", n_train=6_000, n_cal=2_500, n_test=600,
                        delta=0.01, stride=1)
    m1, d1 = fit_reference_detector(env, default_params, jax.random.PRNGKey(0), cfg1, rng=0)
    fpr_stride1 = _alarm_fraction(env, default_params, m1, d1, jax.random.PRNGKey(100), n_seeds, cfg1)

    cfg5 = Phase0Config(env_name="Pendulum-v1", n_train=6_000, n_cal=2_500, n_test=600,
                        delta=0.01, stride=5)
    m5, d5 = fit_reference_detector(env, default_params, jax.random.PRNGKey(0), cfg5, rng=0)
    fpr_stride5 = _alarm_fraction(env, default_params, m5, d5, jax.random.PRNGKey(100), n_seeds, cfg5)

    assert fpr_stride5 <= 0.20, f"stride=5 should control FPR, got {fpr_stride5:.2f}"
    assert fpr_stride5 < fpr_stride1, (
        f"decorrelation should reduce FPR: stride1={fpr_stride1:.2f}, stride5={fpr_stride5:.2f}"
    )
