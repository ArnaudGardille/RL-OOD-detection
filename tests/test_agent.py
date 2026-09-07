"""Part B: the rejax policy beats random, and the reward-drop detector works.

The reward-drop logic is unit-tested on synthetic streams (fast, deterministic); the policy
training is a small smoke check. Requires the ``[agents]`` extra (rejax)."""

import numpy as np
import jax
import pytest

from ood_rl import RewardDropDetector
from ood_rl.experiment import metrics

rejax = pytest.importorskip("rejax")


def test_ppo_beats_random_on_cartpole():
    import gymnax
    from ood_rl.agents.train import train_policy

    policy = train_policy("CartPole-v1", total_timesteps=100_000, num_envs=16, seed=0)
    env, params = gymnax.make("CartPole-v1")

    def episode_return(key, act_fn):
        obs, state = env.reset(key, params)
        total = 0.0
        for _ in range(500):
            key, ka, ks = jax.random.split(key, 3)
            obs, state, r, done, _ = env.step(ks, state, act_fn(obs, ka), params)
            total += float(r)
            if bool(done):
                break
        return total

    rand = lambda o, k: env.action_space(params).sample(k)
    trained_ret = np.mean([episode_return(jax.random.fold_in(jax.random.PRNGKey(1), i), policy) for i in range(10)])
    random_ret = np.mean([episode_return(jax.random.fold_in(jax.random.PRNGKey(2), i), rand) for i in range(10)])
    assert trained_ret > random_ret + 20, f"PPO ({trained_ret:.0f}) not clearly above random ({random_ret:.0f})"


def test_reward_drop_controls_fpr_and_detects():
    rng = np.random.default_rng(0)
    nominal = rng.normal(1.0, 0.1, size=(40, 400))
    det = RewardDropDetector(delta=0.05, seg_len=400)
    det.calibrate(nominal)

    fpr = metrics.detection_rate([det.run(s)[1] for s in rng.normal(1.0, 0.1, size=(20, 400))])
    assert fpr <= 0.25, f"reward-drop FPR not controlled: {fpr:.2f}"

    rate = metrics.detection_rate([det.run(s)[1] for s in rng.normal(0.3, 0.1, size=(20, 400))])
    assert rate >= 0.9, f"reward-drop missed a clear drop: {rate:.2f}"
