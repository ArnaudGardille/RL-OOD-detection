"""Train a pure-JAX policy via rejax (native gymnax), for the reward-based baseline.

The trained policy is exposed behind a thin ``predict(obs, key) -> action`` interface, so
the rest of the framework (and the rollout's ``policy=`` hook) is agnostic to the RL library.
rejax is an optional dependency (``pip install -e ".[agents]"``); it is imported lazily.
"""

import jax


def train_policy(env_name, total_timesteps=200_000, num_envs=16, seed=0, **ppo_kwargs):
    """Train PPO on the nominal env and return a jittable ``predict(obs, key) -> action``.

    The returned callable also carries ``.train_state`` and ``.act`` (the raw rejax act fn).
    """
    from rejax import PPO

    algo = PPO.create(
        env=env_name,
        total_timesteps=total_timesteps,
        eval_freq=total_timesteps,
        num_envs=num_envs,
        **ppo_kwargs,
    )
    train_state, _ = algo.train(jax.random.PRNGKey(seed))
    act = algo.make_act(train_state)  # act(obs, rng) -> action

    def predict(obs, key):
        return act(obs, key)

    predict.act = act
    predict.train_state = train_state
    return predict
