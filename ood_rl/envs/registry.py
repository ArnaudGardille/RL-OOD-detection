"""Environment registry: thin wrapper over ``gymnax.make`` + OOD grid."""

import gymnax

from .params import DEFAULTS, make_ood_grid


def make_env(name):
    """Instantiate a gymnax env and its one-factor OOD grid.

    Returns ``(env, default_params, ood_configs)`` where:
      - ``env.reset(key, params) -> (obs, state)``
      - ``env.step(key, state, action, params) -> (obs, state, reward, done, info)``
      - ``ood_configs`` is the list from :func:`make_ood_grid` (empty if the env has
        no registered physical knobs yet).
    """
    env, default_params = gymnax.make(name)
    ood_configs = make_ood_grid(name, default_params) if name in DEFAULTS else []
    return env, default_params, ood_configs
