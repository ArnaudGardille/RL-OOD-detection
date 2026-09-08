"""Transition collection + reference detection pipeline.

Reimplements the legacy ``Memory`` + ``create_dataset`` (old ``rl_ood.py``) as a jittable,
``vmap``-ready ``jax.lax.scan`` over gymnax steps, preserving the dataset semantics:

  - sliding window of the last ``memory_size`` ``(obs, action)`` pairs, initialized with
    the first observation repeated and zero actions;
  - ``X[t]`` = flattened window after taking the step; ``y[t]`` = ``obs[t] - obs[t-1]``;
  - default uniform-random policy (the dynamics model trains off random transitions).

``params`` flows through the scan as data, so the compiled rollout (a) compiles once per
``(env, n_steps, memory_size, mode)`` and (b) ``vmap``s over a batch of EnvParams to sweep
the whole OOD grid in parallel (see ``ood_rl/experiment/sweep.py``). A non-stationary
variant switches params at a known change-point for online detection-delay measurement.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from gymnax.environments import spaces

from ..models.knn import KNNDynamics
from ..detectors.conformal_martingale import ConformalMartingaleDetector

_ROLLOUT_CACHE = {}


def _action_dim(space) -> int:
    if isinstance(space, spaces.Discrete):
        return 1
    return int(np.prod(space.shape))


def _random_act_fn(env):
    def act(key, obs, params):
        return env.action_space(params).sample(key)
    return act


def _policy_act_fn(policy):
    # policy has signature (obs, key) -> action (matches rejax's make_act act fn).
    def act(key, obs, params):
        return policy(obs, key)
    return act


def _broadcast_params(params, n):
    return jtu.tree_map(lambda x: jnp.broadcast_to(jnp.asarray(x), (n,) + jnp.shape(x)), params)


def _scan_body(env, act_dim, act_fn):
    def body(carry, xs):
        obs, state, ho, ha = carry
        key, params_t = xs
        ak, sk = jax.random.split(key)
        action = act_fn(ak, obs, params_t)
        n_obs, n_state, reward, _done, _info = env.step(sk, state, action, params_t)
        av = jnp.reshape(jnp.asarray(action, jnp.float32), (act_dim,))
        nho = jnp.roll(ho, -1, axis=0).at[-1].set(n_obs)
        nha = jnp.roll(ha, -1, axis=0).at[-1].set(av)
        window = jnp.concatenate([nho, nha], axis=1).reshape(-1)
        delta = n_obs - obs
        return (n_obs, n_state, nho, nha), (window, delta, reward)
    return body


def _init_carry(env, params, key, memory_size, obs_dim, act_dim):
    obs0, state0 = env.reset(key, params)
    ho0 = jnp.broadcast_to(obs0, (memory_size, obs_dim))
    ha0 = jnp.zeros((memory_size, act_dim))
    return obs0, state0, ho0, ha0


def _build_rollout(env, n_steps, memory_size, obs_dim, act_dim, act_fn):
    body = _scan_body(env, act_dim, act_fn)

    def rollout(key, params):
        rkey, skey = jax.random.split(key)
        carry = _init_carry(env, params, rkey, memory_size, obs_dim, act_dim)
        keys = jax.random.split(skey, n_steps)
        params_seq = _broadcast_params(params, n_steps)
        _, (X, y, r) = jax.lax.scan(body, carry, (keys, params_seq))
        return X, y, r

    return jax.jit(rollout)


def _build_nonstationary_rollout(env, n_steps, memory_size, obs_dim, act_dim, act_fn):
    body = _scan_body(env, act_dim, act_fn)

    def rollout(key, params_before, params_after, t_change):
        rkey, skey = jax.random.split(key)
        carry = _init_carry(env, params_before, rkey, memory_size, obs_dim, act_dim)
        keys = jax.random.split(skey, n_steps)
        mask = jnp.arange(n_steps) < t_change  # True before the change-point
        before = _broadcast_params(params_before, n_steps)
        after = _broadcast_params(params_after, n_steps)
        params_seq = jtu.tree_map(
            lambda b, a: jnp.where(mask.reshape((-1,) + (1,) * (b.ndim - 1)), b, a),
            before, after,
        )
        _, (X, y, r) = jax.lax.scan(body, carry, (keys, params_seq))
        return X, y, r

    return jax.jit(rollout)


def _get_rollout(env, n_steps, memory_size, obs_dim, act_dim, policy, nonstationary):
    mode = "random" if policy is None else id(policy)
    tag = "ns" if nonstationary else "st"
    ckey = (id(env), tag, int(n_steps), int(memory_size), obs_dim, act_dim, mode)
    rollout = _ROLLOUT_CACHE.get(ckey)
    if rollout is None:
        act_fn = _random_act_fn(env) if policy is None else _policy_act_fn(policy)
        builder = _build_nonstationary_rollout if nonstationary else _build_rollout
        rollout = builder(env, int(n_steps), int(memory_size), obs_dim, act_dim, act_fn)
        _ROLLOUT_CACHE[ckey] = rollout
    return rollout


def collect_transitions(env, params, key, n_steps, memory_size=10, stride=1,
                        policy=None, return_rewards=False):
    """Roll out a (random or policy) actor and return ``(X, y)`` numpy arrays.

    ``X`` is ``(n_steps, memory_size * (obs_dim + act_dim))``; ``y`` is ``(n_steps, obs_dim)``.
    ``stride > 1`` subsamples to reduce temporal correlation. If ``return_rewards`` is set,
    also returns the per-step reward stream.
    """
    obs_dim = int(np.prod(env.observation_space(params).shape))
    act_dim = _action_dim(env.action_space(params))
    rollout = _get_rollout(env, n_steps, memory_size, obs_dim, act_dim, policy, nonstationary=False)

    X, y, r = rollout(key, params)
    X, y, r = np.asarray(X), np.asarray(y), np.asarray(r)
    if stride > 1:
        X, y, r = X[::stride], y[::stride], r[::stride]
    return (X, y, r) if return_rewards else (X, y)


def collect_nonstationary(env, params_before, params_after, t_change, key, n_steps,
                          memory_size=10, stride=1, policy=None):
    """Roll out with a dynamics change-point at ``t_change``.

    Returns ``(X, y, rewards, change_index)`` where ``change_index`` is the change-point
    expressed in (possibly strided) sample units — the ground truth for detection delay.
    """
    obs_dim = int(np.prod(env.observation_space(params_before).shape))
    act_dim = _action_dim(env.action_space(params_before))
    rollout = _get_rollout(env, n_steps, memory_size, obs_dim, act_dim, policy, nonstationary=True)

    X, y, r = rollout(key, params_before, params_after, jnp.asarray(int(t_change)))
    X, y, r = np.asarray(X), np.asarray(y), np.asarray(r)
    if stride > 1:
        X, y, r = X[::stride], y[::stride], r[::stride]
    change_index = int(t_change // stride)
    return X, y, r, change_index


# --- detection pipeline glue (shared calibration across all detectors) ----------------

def fit_world_model(env, params, key, cfg):
    """Train the KNN dynamics world-model on nominal random transitions."""
    X_tr, y_tr = collect_transitions(env, params, key, cfg.n_train, cfg.memory_size, cfg.stride)
    return KNNDynamics(n_neighbors=cfg.n_neighbors).fit(X_tr, y_tr)


def calibrate_detector(detector, model, env, params, key, cfg):
    """Calibrate any Detector on the shared in-distribution nonconformity stream."""
    X_cal, y_cal = collect_transitions(env, params, key, cfg.n_cal, cfg.memory_size, cfg.stride)
    detector.calibrate(model.residual(X_cal, y_cal))
    return detector


def fit_reference_detector(env, params, key, cfg, rng=None):
    """Convenience: train world-model + calibrate the conformal-martingale detector."""
    k_train, k_cal = jax.random.split(key)
    model = fit_world_model(env, params, k_train, cfg)
    detector = ConformalMartingaleDetector(epsilon=cfg.epsilon, delta=cfg.delta, rng=rng)
    calibrate_detector(detector, model, env, params, k_cal, cfg)
    return model, detector


def detect(model, detector, X, y):
    """Score a transition stream and return ``(scores, alarm_at)``."""
    alphas = model.residual(X, y)
    return detector.run(alphas)
