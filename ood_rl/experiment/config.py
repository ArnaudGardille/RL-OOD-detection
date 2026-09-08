"""Phase 0 configuration + seeding helpers."""

from dataclasses import dataclass

import jax


@dataclass
class Phase0Config:
    env_name: str = "CartPole-v1"
    seed: int = 0
    # dataset sizes (transitions)
    n_train: int = 10_000   # train the dynamics world-model
    n_cal: int = 3_000      # conformal calibration set (in-distribution)
    n_test: int = 1_000     # streamed test sequence
    # rollout
    memory_size: int = 10   # sliding (obs, action) history window
    stride: int = 1         # temporal subsampling (decorrelation knob)
    # world-model
    n_neighbors: int = 5
    # detector
    epsilon: float = 0.92   # power-martingale exponent
    delta: float = 0.01     # target false-alarm level (Ville threshold = log(1/delta))


def make_key(seed: int):
    """Explicit PRNG key — reproducibility the legacy code entirely lacked."""
    return jax.random.PRNGKey(seed)
