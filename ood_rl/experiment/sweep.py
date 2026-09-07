"""Parallel OOD-grid sweep via ``vmap``, plus stream-level detector calibration.

This is the JAX payoff: the whole one-factor OOD grid (and the batch of nominal calibration
streams) is rolled out in parallel on one device (validated at ~2.5 ms / 16 configs x 200
steps). Detection scoring afterwards is numpy (the streams are small).
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from .rollout import _get_rollout, _action_dim, fit_world_model

_SWEEP_CACHE = {}


def sweep_streams(env, configs, key, cfg):
    """Roll out every config's params in parallel.

    Returns ``(X[B, T, .], y[B, T, .], rewards[B, T])`` as numpy arrays, one row per config.
    """
    params_list = [c["params"] for c in configs]
    batched = jtu.tree_map(lambda *xs: jnp.stack(xs), *params_list)
    B = len(configs)
    p0 = params_list[0]
    obs_dim = int(np.prod(env.observation_space(p0).shape))
    act_dim = _action_dim(env.action_space(p0))

    ckey = (id(env), int(cfg.n_test), int(cfg.memory_size), obs_dim, act_dim)
    vroll = _SWEEP_CACHE.get(ckey)
    if vroll is None:
        roll = _get_rollout(env, cfg.n_test, cfg.memory_size, obs_dim, act_dim,
                            policy=None, nonstationary=False)
        vroll = jax.jit(jax.vmap(roll, in_axes=(0, 0)))
        _SWEEP_CACHE[ckey] = vroll

    keys = jax.random.split(key, B)
    X, y, r = vroll(keys, batched)
    return np.asarray(X), np.asarray(y), np.asarray(r)


def alpha_streams(model, X, y, stride=1):
    """Per-config nonconformity streams ``alpha[B, T']`` from batched ``(X, y)``."""
    B, T = X.shape[0], X.shape[1]
    alphas = model.residual(X.reshape(B * T, -1), y.reshape(B * T, -1)).reshape(B, T)
    return alphas[:, ::stride] if stride > 1 else alphas


def nominal_alpha_streams(env, params, key, model, cfg, n_streams=200):
    """A batch ``[n_streams, T']`` of in-distribution nonconformity streams."""
    X, y, _ = sweep_streams(env, [{"params": params}] * n_streams, key, cfg)
    return alpha_streams(model, X, y, stride=cfg.stride)


def fit_detectors(env, params, key, cfg, factories, n_cal_streams=200):
    """Train the world-model and calibrate every detector on the same nominal streams.

    ``factories`` maps a name to a zero-arg constructor (so each call gets a fresh detector).
    Returns ``(model, {name: calibrated_detector})``.
    """
    k_train, k_cal = jax.random.split(key)
    model = fit_world_model(env, params, k_train, cfg)
    cal = nominal_alpha_streams(env, params, k_cal, model, cfg, n_cal_streams)
    detectors = {name: factory() for name, factory in factories.items()}
    for det in detectors.values():
        det.calibrate(cal)
    return model, detectors


def run_detector_on_streams(detector, streams):
    """Run a calibrated detector over each row; return list of ``(alarm_at, sup_score)``."""
    results = []
    for s in streams:
        scores, alarm_at = detector.run(s)
        results.append((alarm_at, float(np.max(scores)) if len(scores) else float("nan")))
    return results
