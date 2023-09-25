from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
env_id = "Pendulum-v1"
env = make_vec_env(env_id, n_envs=1)

# Instantiate the agent
model = PPO(
    "MlpPolicy",
    env,
    gamma=0.98,
    # Using https://proceedings.mlr.press/v164/raffin22a.html
    use_sde=True,
    sde_sample_freq=4,
    learning_rate=1e-3,
    verbose=1,
)

from rl_ood import *

# Train the agent
#for i in range(3, 6):
steps = int(10**5)
print("steps", steps)
model.learn(total_timesteps=steps, progress_bar=True)
mean, std = evaluate(env, model, nb_episodes=10, render=True)
print("mean", mean )
print("std", std )
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print("mean", mean )
print("std", std )