import numpy as np
import jax

from ood_rl import make_env, Phase0Config, KNNDynamics, ConformalMartingaleDetector
from ood_rl.experiment.rollout import collect_transitions, fit_reference_detector, detect


def test_make_env_and_rollout_shapes():
    env, params, ood_configs = make_env("CartPole-v1")
    X, y = collect_transitions(env, params, jax.random.PRNGKey(0), n_steps=200, memory_size=10)
    obs_dim, act_dim = 4, 1
    assert X.shape == (200, 10 * (obs_dim + act_dim))
    assert y.shape == (200, obs_dim)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    # one-factor OOD grid present: 5 knobs x 21 scales
    assert len(ood_configs) == 5 * 21
    assert {c["change"] for c in ood_configs} == {
        "gravity", "masscart", "masspole", "length", "force_mag",
    }


def test_detector_cycle_runs_and_is_finite():
    env, params, _ = make_env("CartPole-v1")
    cfg = Phase0Config(n_train=2_000, n_cal=1_000, n_test=300)
    model, detector = fit_reference_detector(env, params, jax.random.PRNGKey(1), cfg, rng=0)
    X, y = collect_transitions(env, params, jax.random.PRNGKey(2), cfg.n_test, cfg.memory_size)
    scores, alarm_at = detect(model, detector, X, y)
    assert scores.shape[0] == X.shape[0]
    assert np.isfinite(scores).all()
    assert isinstance(alarm_at, int) and alarm_at >= -1


def test_conformal_pvalues_are_uniform_under_calibration():
    """Sanity on the core math: conformal p-values on fresh in-distribution scores are
    ~Uniform(0,1), so their mean is ~0.5 (this is what gives FPR control)."""
    rng = np.random.default_rng(0)
    cal = rng.normal(size=5000)
    det = ConformalMartingaleDetector(rng=1)
    det.calibrate(cal)
    test = rng.normal(size=5000)  # same distribution
    ps = np.array([det._p_value(float(a)) for a in test])
    assert 0.45 < ps.mean() < 0.55
