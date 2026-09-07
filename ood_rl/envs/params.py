"""Physical-knob <-> gymnax EnvParams mapping and OOD-grid generation.

Mirrors the legacy ``CARTPOLE_VALUES`` / ``instanciate_cartpole`` /
``get_ood_configs`` logic (old ``rl_ood.py``) but functionally, on gymnax
``EnvParams`` (which are immutable flax structs updated via ``.replace``).
"""

import numpy as np

# The knobs we perturb, keyed by gymnax EnvParams field names (verified against
# gymnax classic_control source).
DEFAULTS = {
    "CartPole-v1": {
        "gravity": 9.8,
        "masscart": 1.0,
        "masspole": 0.1,
        "length": 0.5,
        "force_mag": 10.0,
    },
    "Pendulum-v1": {
        "g": 10.0,
        "m": 1.0,
        "l": 1.0,
        "max_speed": 8.0,
        "max_torque": 2.0,
    },
}


def build_params(env_name, base_params, **overrides):
    """Return a copy of ``base_params`` with physical-knob overrides applied.

    gymnax stores *precomputed* derived fields for CartPole (``total_mass`` and
    ``polemass_length``). If we change ``masscart`` / ``masspole`` / ``length`` we
    must recompute them, otherwise the simulated dynamics are inconsistent — this
    is exactly what the legacy ``instanciate_cartpole`` did by hand.
    """
    params = base_params.replace(**overrides)
    if env_name == "CartPole-v1":
        masscart = float(params.masscart)
        masspole = float(params.masspole)
        length = float(params.length)
        params = params.replace(
            total_mass=masscart + masspole,
            polemass_length=masspole * length,
        )
    return params


def make_ood_grid(env_name, base_params, num=21, span=(-1.0, 1.0)):
    """One-factor-at-a-time OOD configs.

    For each knob, scale the default value by ``logspace(span, num)`` (the midpoint
    scale 1.0 reproduces the nominal env — a useful in-distribution control). Each
    config differs from the default by a single factor, like the legacy benchmark.

    Returns a list of dicts: ``{"change", "value", "scale", "params"}``.
    """
    defaults = DEFAULTS[env_name]
    scales = np.logspace(span[0], span[1], num=num)
    configs = []
    for knob, default in defaults.items():
        for scale in scales:
            value = float(default * scale)
            params = build_params(env_name, base_params, **{knob: value})
            configs.append(
                {"change": knob, "value": value, "scale": float(scale), "params": params}
            )
    return configs
