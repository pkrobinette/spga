"""
CartPole environment from OpenAI Gym adjusted for action masking.

Resources:
-----------
https://github.com/PacktPublishing/Mastering-Reinforcement-Learning-with-Python/blob/master/Chapter10/custom_mcar.py

"""


import gym
from gym.spaces import Box, Dict
import numpy as np


class CartPole(gym.Env):
    def __init__(self, env_config={}):
        #
        # set up proxy environment
        #
        self.env = gym.make("CartPole-v0") # could use version v1
        if env_config.get("seed", 0):
            self.env.seed(env_config["seed"])
            print("Using Seed: ", env_config["seed"])
        self.action_space = self.env.action_space
        self.t = 0   # steps
        self.x_threshold = env_config.get("x_threshold", 1.5) # boundary
        self.use_action_masking = env_config.get("use_action_masking", False) 
        self.action_mask = None
        # extend the initial start conditions. Not used in this work.
        self.extend = env_config.get("extend", False)
        self.reset()
        if self.use_action_masking: # --> could probably change this
            self.observation_space = Dict(
                {
                    "action_mask": Box(0, 1, shape=(self.action_space.n,)),
                    "actual_obs": self.env.observation_space,
                }
            )
        else:
            self.observation_space = self.env.observation_space
        
            
    def reset(self, init = False, state=None):
        """
        Reset the environemnt
        """
        raw_obs = None
        # if extend, extend the initial conditions
        if self.extend:
            fluff = self.env.reset()
            raw_obs = np.array([np.random.uniform(low=-self.x_threshold/3, high=self.x_threshold/3),
                       np.random.uniform(low=-0.05, high=0.05),
                       np.random.uniform(low=-0.10, high=0.10),
                       np.random.uniform(low=-0.05, high=0.05)])
            
            for i in range(len(self.env.state)):
                self.env.state[i] = raw_obs[i]
            
        else:
            raw_obs = self.env.reset()
        if init:
            for i in range(len(self.env.state)):
                self.env.state[i] = state[i]
            raw_obs = state
        self.t = 0
        if self.use_action_masking:
            self.update_avail_actions(raw_obs)
            obs = {
                "action_mask": self.action_mask,
                "actual_obs": raw_obs,
            }
        else:
            obs = raw_obs
        
        return obs
    
    def render(self):
        """ Render the environment """
        self.env.render()
        

    def step(self, action):
        """
        Take a step in the environment
        """
        self.t += 1
        state, reward, done, info = self.env.step(action)
        if self.use_action_masking:
            self.update_avail_actions(state)
            obs = {
                "action_mask": self.action_mask,
                "actual_obs": state,
            }
        else:
            obs = state
            
        if self.t >= 200: # ------ change if using v1
            done = True
        return obs, reward, done, info

    def update_avail_actions(self, obs):
        """
        Update available actions in the action mask
        """
        self.action_mask = np.array([1.0] * self.action_space.n)
        # observation space
        #0: Cart Position
        #1: cart velocity
        #2: pole angle
        #3: pole angular velocity
        pos, vel, theta, theta_vel = obs
        
        # 0: push left, 1: push right
        # TO DO: Position and Velocity Constraints
        
        # only allow right action
        if pos <= -self.x_threshold:
            self.action_mask[0] = 0
        # only allow left action
        if pos >= self.x_threshold:
            self.action_mask[1] = 0