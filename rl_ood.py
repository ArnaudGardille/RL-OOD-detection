import gym #gymnasium as
#import gymnasium as gym
#import gym
from gym.wrappers import TimeLimit
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from itertools import product
from stable_baselines3 import A2C, DQN, DDPG
from pathlib import Path
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import copy
from tqdm import trange
import os
import pandas as pd
import joblib
from copy import copy
from tqdm import tqdm, trange
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
#from skmultiflow.lazy import KNNRegressor
#from skmultiflow.meta.multi_output_learner import MultiOutputLearner
import scipy.integrate as integrate
from scipy import stats
from scipy import integrate
import pygame

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("error", category=RuntimeWarning)

CARTPOLE_VALUES = {
    'Gravity' : 9.8,
    'Mass_cart' : 1.0,
    'Length_pole' : 0.5,
    'Mass_pole' : 0.1,
    'Force_magnitude' : 10.0,
}
PENDULUM_VALUES = {
    'Gravity': 10.0,
    'Mass_pole': 1.0,
    'Length_pole': 1.0,
    'Max_speed': 8.0,
    'Max_torque': 2.0,
}
MOUNTAIN_CAR_VALUES = {
    'Gravity': 0.0025,
    'Force': 0.001,
}

path = Path.cwd()
device = 'cpu'

class TransformReward(gym.RewardWrapper):
    def __init__(self, env: gym.Env, f):
        super().__init__(env)
        assert callable(f)
        self.f = f

    def reward(self, reward):
        return self.f(reward)

class ObjectiveReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs,reward, terminated, info = self.env.step(action)
        reward = float(terminated)
        return obs, reward, terminated, info

def create_ood_values(default_values, n=5,):
    results = {}
    for k, v, in default_values.items():
        results[k] = v*np.logspace(-1, 1, num=21)
        #results[k] = np.delete(v*np.logspace(-1, 1, num=21), 10)
        #results[k] = [v*2**i for i in range(-n, 0)] + [v*2**i for i in range(1, n+1)]
    return results
    

def get_cartpole_values():
    default_values = {}
    values = {}

    default_values['Gravity'] = 9.8
    values['Gravity'] = [0.98, 1.09, 1.23, 1.4, 1.63, 1.96, 2.45, 3.27, 4.9, 19.6, 29.4, 39.2, 49.0, 58.8, 68.6, 78.4, 88.2, 98.0]

    default_values['Mass_cart'] = 1.0
    values['Mass_cart'] = [0.1, 0.1111, 0.125, 0.1429, 0.1667, 0.2, 0.25, 0.3333, 0.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    default_values['Length_pole'] = 0.5
    values['Length_pole'] = [0.05, 0.0556, 0.0625, 0.0714, 0.0833, 0.1, 0.125, 0.1667, 0.25, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    default_values['Mass_pole'] = 0.1
    values['Mass_pole'] = [0.01, 0.0111, 0.0125, 0.0143, 0.0167, 0.02, 0.025, 0.0333, 0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    default_values['Force_magnitude'] = 10.0
    values['Force_magnitude'] = [1.0, 1.1111, 1.25, 1.4286, 1.6667, 2.0, 2.5, 3.3333, 5.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    return default_values, values

def instanciate_cartpole(config, limit=200, render_mode=None):
    #env = gym.make("CartPole-v1").env
    env = gym.make("CartPole-v1").env
    env.gravity = config['Gravity']
    env.masscart = config['Mass_cart']
    env.masspole = config['Mass_pole']
    env.total_mass = env.masspole + env.masscart
    env.length =  config['Length_pole'] # actually half the pole's length
    env.polemass_length = env.masspole * env.length
    env.force_mag = config['Force_magnitude']
    env = TimeLimit(env, limit)
    return env


def get_pendulum_values():
    default_values = {}
    values = {}

    default_values['Gravity'] = 10.0
    values['Gravity'] = [0.5, 1.0, 2.0, 5.0, 20.0, 50.0, 100.0, 200.0]

    default_values['Mass_pole'] = 1.0
    values['Mass_pole'] = [0.05, 0.1, 0.2, 0.5, 2.0, 5.0, 10.0, 20.0]

    default_values['Length_pole'] = 1.0
    values['Length_pole'] = [0.05, 0.1, 0.2, 0.5, 2.0, 5.0, 10.0, 20.0]
    
    default_values['Max_speed'] = 8.0
    values['Max_speed'] = [0.4, 0.8, 1.6, 4.0, 16.0, 40.0, 80.0, 160.0]

    default_values['Max_torque'] = 2.0
    values['Max_torque'] = [0.1, 0.2, 0.4, 1.0, 4.0, 10.0, 20.0, 40.0]

    return default_values, values


def instanciate_pendulum(config, limit=200, render_mode=None):
    env = gym.make("Pendulum-v1").env
    env.max_speed = config['Max_speed']
    env.max_torque = config['Max_torque']
    env.g = config['Gravity']
    env.m = config['Mass_pole']
    env.l = config['Length_pole']
    env = TimeLimit(env, limit)
    env = TransformReward(env, lambda x:(x+17)/17)
    return env


def get_mountain_car_values():
    default_values = {}
    values = {}

    default_values['Gravity'] = 0.0025
    values['Gravity'] = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.07, 0.1]

    default_values['Force'] = 0.001
    values['Force'] = [0.0001, 0.0005, 0.005, 0.01, 0.05, 0.1]

    return default_values, values

def instanciate_mountain_car(config, limit=200, render_mode=None):
    env = gym.make("MountainCar-v0").env
    env.force = config['Force']
    env.gravity = config['Gravity']
    env = TimeLimit(env, limit)
    env = ObjectiveReward(env)
    return env

def instanciate_mountain_car_continuous(config, limit=200, render_mode=None):
    env = gym.make("MountainCarContinuous-v0").env
    env.force = config['Force']
    env.gravity = config['Gravity']
    env = TimeLimit(env, limit)
    env = ObjectiveReward(env)
    return env


def get_possible_combinaisons(values):
    return [x for x in product(*list(values.values()))]

def get_ood_configs(default_values, values):
    """
    Gives ood config that differ form the defalut config by only one value
    """
    ood_configs = []
    #changes = []
    for key in values:
        
        for value in values[key]:
            ood_config = copy(default_values)
            #if value != default_values[key]:
            ood_config[key] = value
            ood_config['change'] = key
            ood_configs.append(ood_config)
                #changes.append({key:value})

    return ood_configs #, changes




def evaluate(env, agent, nb_episodes=1000, render=False):
    total_rewards = []
    
    for ep in range(nb_episodes):
        total_reward = 0.0
        #observation, _ = env.reset()
        observation = env.reset()
        terminated = False
        
        while terminated is False:
            action, _state = agent.predict(observation, deterministic=True)
            #action = env.action_space.sample()
            observation, reward, terminated, info = env.step(action)
            #observation, reward, terminated, truncated, info = env.step(action)
            #terminated = terminated or truncated
            total_reward += reward

            if render:
                env.render()
                
        total_rewards.append(total_reward)

    env.close()
    
    return np.mean(total_rewards), np.std(total_rewards)

def get_space_limits(space):
    if isinstance(space, gym.spaces.Discrete):
        return np.array([0.0], dtype=np.float32),  np.array([float(space.n -1)], dtype=np.float32)
    else:
        return space.low, space.high

class Memory(gym.Wrapper):
    def __init__(self, env, size, verbose=False, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.size = size
        self.verbose = verbose
        
        self.obs_limits = get_space_limits(env.observation_space)  
        self.act_limits = get_space_limits(env.action_space)
        
        self.state_size = self.obs_limits[0].shape[0]
        self.action_size = self.act_limits[0].shape[0]
       
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        #done = done or trunc
        
        self.history_obs = np.roll(self.history_obs, -self.state_size)
        self.history_obs[-1] = obs
        
        self.history_action = np.roll(self.history_action, -self.action_size)
        self.history_action[-1] = action

        if self.verbose:
            print(self.history_obs)
            print(self.history_action)
        
        return obs, reward, done, False, info

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        self.history_obs = np.full((self.size, self.state_size), observation)
        self.history_action = np.full((self.size, self.action_size), 0)
        return observation, None
        

    def get_history(self, concat=False):
        if concat:
            return np.concatenate((self.history_obs, self.history_action), axis=1)
        else:
            return self.history_obs, self.history_action
    


def create_dataset(env, nb_steps = 10000, memory_size = 10, verbose=False):
    
    obs_limits = get_space_limits(env.observation_space)  
    act_limits = get_space_limits(env.action_space)
    env = Memory(env, memory_size)
    state_size = env.obs_limits[0].shape[0]
    action_size = env.act_limits[0].shape[0]
    
    input_size = memory_size*(state_size+action_size)

    X = np.zeros((nb_steps, input_size))
    y = np.zeros((nb_steps, state_size))

    
    observation = env.reset()
    for t in range(nb_steps):

        action = env.action_space.sample()
        previous_obs = observation
        observation, reward, terminated, info = env.step(action)
        #terminated = terminated or truncated
        history = env.get_history(True).reshape(input_size)

        real_diff = np.array(observation-previous_obs)

        X[t] = history
        y[t] = real_diff

        if terminated:
            observation = env.reset()

    return X, y


def compute_p_values(X):
    """
    p_value for the normal distribution
    X is supposed to be normalized
    """
    t = - np.abs(X)
    p_value = 2.0 * stats.norm.cdf(t)
    return p_value

def martingale(p_values):
        func = lambda x : np.prod(x * (p_values ** (x-1)))
        result = integrate.quad(func, 0, 1)
        return result[0]



class MartingaleOODDetector():
    def __init__(self, env: gym.Env, verbose=False, *args, **kwargs) -> None:

        self.verbose = verbose

        # training the model
        X_pred, y_pred = create_dataset(env, nb_steps=10000)
        self.pred_model = MultiOutputRegressor(KNeighborsRegressor()).fit(X_pred, y_pred)
        #self.conf_model = conf_model

        self.in_distrib_score = self.test_ood(env)

        if self.verbose:
            print("Anomaly score of the training distribution: ", self.in_distrib_score)

    def get_in_distrib_score(self):
        return self.in_distrib_score

    def test_ood(self, env, nb_steps=1000):
        """
        Compute the ood score
        """
        X_val, y_val = create_dataset(env, nb_steps)
        errors = np.abs((self.pred_model.predict(X_val) - y_val))

        if self.verbose:
            print("Absolute error")
            print("Mean: ", errors.mean())
            print("Std: ", errors.std())
            print()


        # Calibration of the ood detector
        pre_ood_score = martingale(compute_p_values(errors))   
        ood_score = nb_steps * np.log(1 + pre_ood_score) #/nb_steps
        #print("corrected score ", np.log10(ood_score)/nb_steps)
        return ood_score

    def stop_above_threshold(self, env, threshold, start_at=50, nb_steps=1000):
        """
        Compute the number of step for the ood score to go above the threshold
        """
        X_val, y_val = create_dataset(env, nb_steps)

        for step in range(start_at,nb_steps):
            errors = np.abs((self.pred_model.predict(X_val[:step]) - y_val[:step]))

            if self.verbose:
                print("Absolute error")
                print("Mean: ", errors.mean())
                print("Std: ", errors.std())
                print()


            # Calibration of the ood detector
            pre_ood_score = martingale(compute_p_values(errors))   
            ood_score = step * np.log(1 + pre_ood_score) #/nb_steps
            #print("corrected score ", np.log10(ood_score)/nb_steps)
            if ood_score > threshold:
                return step, ood_score
        return nb_steps, ood_score

    def save(self, path):
        np.save(path / 'nonconformity_scores.npy', self.nonconformity_scores)
        
    def load(self, path):
        self.nonconformity_scores = np.load(path / 'nonconformity_scores.npy')



