import numpy as np
import jax

from ood_rl import make_env, Phase0Config, ConformalMartingaleDetector
from ood_rl.experiment import sweep


def test_vmap_sweep_covers_grid_and_separates_strong_ood():
    cfg = Phase0Config("CartPole-v1", n_train=4_000, n_test=400, delta=0.01, stride=1)
    env, dp, configs = make_env(cfg.env_name)
    k_fit, k_sweep = jax.random.split(jax.random.PRNGKey(0))

    model, dets = sweep.fit_detectors(
        env, dp, k_fit, cfg,
        {"conformal": lambda: ConformalMartingaleDetector(delta=cfg.delta, rng=0)},
        n_cal_streams=40,
    )

    X, y, r = sweep.sweep_streams(env, configs, k_sweep, cfg)
    assert X.shape == (len(configs), cfg.n_test, 10 * (4 + 1))
    assert r.shape == (len(configs), cfg.n_test)

    alphas = sweep.alpha_streams(model, X, y, stride=cfg.stride)
    assert alphas.shape == (len(configs), cfg.n_test)
    assert np.isfinite(alphas).all()

    # strong shifts should produce higher conformal statistics than near-nominal configs
    grid = sweep.run_detector_on_streams(dets["conformal"], alphas)
    sup = np.array([g[1] for g in grid])
    scales = np.array([c["scale"] for c in configs])
    strong = np.abs(np.log10(scales)) >= 0.5
    near = np.abs(np.log10(scales)) < 1e-9
    assert sup[strong].mean() > sup[near].mean()
