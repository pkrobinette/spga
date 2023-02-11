"""
RTA Knapsack environment from OpenAI Gym adjusted for action masking.

Resources:
-----------
https://github.com/PacktPublishing/Mastering-Reinforcement-Learning-with-Python/blob/master/Chapter10/custom_mcar.py

"""


import or_gym
from or_gym.utils import create_env
from gym.spaces import Box, Dict
import numpy as np
import math
import gym
from ray import tune

def register_env(env_name, env_config={}):
    env = create_env(env_name)
    tune.register_env(env_name, 
        lambda env_name: env(env_name,
            env_config=env_config))


class Knapsack(gym.Env):
    def __init__(self, env_config={}):
        rt = env_config.pop("runtime", False)
        self.env = or_gym.make("Knapsack-v0", env_config=env_config) # might use version v1
        env_config['runtime'] = rt
        self.observation_space = self.env.observation_space
        self.item_weights = self.env.item_weights
        self.item_values = self.env.item_values
        self.action_space = self.env.action_space
        self.current_weight = self.env.current_weight
        self.max_weight = self.env.max_weight
        self.use_run_time_assurance = env_config.get("runtime", False)
        if self.use_run_time_assurance:
            print("Using RTA ...\n")
        self.reset()
        # if env_config.get("seed", 0):
        #     self.env.seed(env_config["seed"])
        #     print("Using Seed: ", env_config["seed"])
        self.avail_actions = np.where(self.current_weight + self.item_weights > self.max_weight,
            0, 1)
            
            
    def reset(self, init = False, state=None):
        obs = self.env.reset()
        
        return obs
    
    def render(self):
        self.env.render()
        

    def step(self, action):
        pen = 0
        if self.use_run_time_assurance:
            probe_weight, unsafe = self.probe_step(action)
            if unsafe:
                avail_actions = np.where(self.current_weight + self.item_weights > self.max_weight,
            0, 1)
                actions = [i for i in range(5) if avail_actions[i] == 1]
                action = np.random.choice(actions)
                pen = -5
            
        obs, reward, done, info = self.env.step(action)
        
        return obs, reward+pen, done, info
    
    
    def probe_step(self, action):
        """
        Probe step using environment dynamics. Checks if the proposed action will lead
        to an unsafe state.
        """
        probe_weight = self.env.current_weight + self.env.item_weights[action]
        
        unsafe = probe_weight > self.env.max_weight

        return probe_weight, unsafe