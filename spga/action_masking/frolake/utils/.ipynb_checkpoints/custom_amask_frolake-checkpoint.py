"""
Gridworld/frozenLake environment from OpenAI Gym adjusted for action masking.

Resources:
-----------


"""


import gym
from gym.spaces import Box, Dict
import numpy as np
from utils.custom_amask_biggerlake import BiggerLakeEnv
from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

from gym import Env, spaces, utils


MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
    "16x16": [
        "SFFFFFFFFFFFFFFH",
        "FFFFFFFFHHFFFFFF",
        "FFFHFFFFFHFFFHFF",
        "HFFFFFFHFFFFFHFF",
        "FFFFHFFFFHFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFHFFFFFFFFFFH",
        "HFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFH",
        "FFFFFFFFHHFFFFFF",
        "FFFHFFFFFHFFFHFF",
        "HFFFFFFHFFFFFHFF",
        "FFFFHFFFFHFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFHFFFFFFFFFFH",
        "HFFFFFFFFFFFFFFG", 
    ],
    "32x32": [
        "SFFFFFFFFFFFFFFHFFFFFFHFFFFFFFFH",
        "FFFFFHHFFFFFFHFFFFFFHFFFFFFFFFFH",
        "FFFHFFFFFFHFFFFFFFFFFFFFFFFFFHFF",
        "HFFFFFFHFFFFFFFFFFHFFFFFFFFFFFFF",
        "FHFFFFFFFFFFHFFFFFFFFFFFHFFFFFFF",
        "FFFFFHHFFFFFFHFFFFFFHFFFFFFFFFFH",
        "FFFHFFFFFFHFFFFFFFFFFFFFFFFFFHFF",
        "HFFFFFFHFFFFFFFFFFHFFFFFFFFFFFFF",
        "FHFFFFFFFFFFHFFFFFFFFFFFHFFFFFFF",
        "FFFFFHHFFFFFFHFFFFFFHFFFFFFFFFFH",
        "FFFHFFFFFFHFFFFFFFFFFFFFFFFFFHFF",
        "HFFFFFFHFFFFFFFFFFHFFFFFFFFFFFFF",
        "FHFFFFFFFFFFHFFFFFFFFFFFHFFFFFFF",
        "FFFFFHHFFFFFFHFFFFFFHFFFFFFFFFFH",
        "FFFHFFFFFFHFFFFFFFFFFFFFFFFFFHFF",
        "HFFFFFFHFFFFFFFFFFHFFFFFFFFFFFFF",
        "FHFFFFFFFFFFHFFFFFFFFFFFHFFFFFFF",
        "FFFFFHHFFFFFFHFFFFFFHFFFFFFFFFFH",
        "FFFHFFFFFFHFFFFFFFFFFFFFFFFFFHFF",
        "HFFFFFFHFFFFFFFFFFHFFFFFFFFFFFFF",
        "FHFFFFFFFFFFHFFFFFFFFFFFHFFFFFFF",
        "FFFFFHHFFFFFFHFFFFFFHFFFFFFFFFFH",
        "FFFHFFFFFFHFFFFFFFFFFFFFFFFFFHFF",
        "HFFFFFFHFFFFFFFFFFHFFFFFFFFFFFFF",
        "FHFFFFFFFFFFHFFFFFFFFFFFHFFFFFFF",
        "FFFFFFFFHFFFFFFFFFFHFFFFFFFHHFFF",
        "HHHHFFFFFFFFFFFFFHFHFFFFFFFFFFFH",
        "FFFFFFFFHFFFFHFFFFHFHFHFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFH",
        "FFFFFFFFHFFFFHFFFFHFHFHFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFG",
        
    ],
    
#     "64x64": [
        
        
        
#     ],
    
#     "128x128":[
        
        
        
#     ],
}


class FrozenLake(gym.Env):
    def __init__(self, env_config={}):
        # is_slippery = False means deterministic environment
        self.map_name = env_config.get("map_name", "4x4")
        print("Using the {} map".format(self.map_name))
        self.testing = env_config.get("test_mode", False)
        if self.map_name in ["4x4", "8x8"]:
            self.env = gym.make("FrozenLake-v1", is_slippery=False, map_name=self.map_name) #env = gym.make("FrozenLake8x8-v0")
        else:
            self.env = BiggerLakeEnv(is_slippery=False, map_name=self.map_name)
        if env_config.get("seed", 0):
            self.env.seed(env_config["seed"])
            print("Using Seed: ", env_config["seed"])
        self.action_space = self.env.action_space
        self.t = 0
        # self.x_threshold = env_config.get("x_threshold", 1.5)
        self.use_action_masking = env_config.get("use_action_masking", False)
        self.use_run_time_assurance = env_config.get("use_run_time_assurance", False)
        self.action_mask = None
        
        # set up available actions and holes
        if self.map_name == "4x4":
            self.avail_actions, self.holes = self.get_avail_actions("4x4")
            
        else:
            self.avail_actions, self.holes = self.get_avail_actions("8x8")
        
        self.reset()
        if self.use_action_masking: # --> could probably change this
            self.observation_space = Dict(
                {
                    "action_mask": Box(0, 1, shape=(self.action_space.n,)),
                    "state": self.env.observation_space,
                }
            )
        else:
            self.observation_space = self.env.observation_space
            
            
    def get_avail_actions(self, map_name = "4x4"):
        """ 
        Get available actions for the 4x4 map 
        
        Parameters
        ----------
        map_name : str
            "4x4" or "8x8"
            
        Returns
        ---------
        avail_actions : dict
            dictionary of valid actions for each state in the grid
        """
        
        avail_actions = {}
        holes = []
        map_grid = MAPS[self.map_name]
        
        lin_map = []
        nrows = len(map_grid)
        
        for current_row in range(nrows):
            for current_col in range(nrows):
                lin_map.append(map_grid[current_row][current_col])
        
        for current_row in range(nrows):
            for current_col in range(nrows):
                s = current_row*nrows + current_col
                if lin_map[s] == "H":
                    holes.append(s)
                    
                avail_actions[s] = []
                
                # first column
                if s%nrows == 0:
                    if s == 0:
                        #right
                        if lin_map[s + 1]!= "H":
                            avail_actions[s].append(2)
                        # down
                        if lin_map[s + nrows]!= "H":
                            avail_actions[s].append(1)
                        
                    elif s == (nrows*(nrows-1)):
                        # right 
                        if lin_map[s + 1]!= "H":
                            avail_actions[s].append(2)
                        # up
                        if lin_map[s - nrows]!= "H":
                            avail_actions[s].append(3)
                    else:
                        # right
                        if lin_map[s+1]!= "H":
                            avail_actions[s].append(2)
                        # up
                        if lin_map[s-nrows]!= "H":
                            avail_actions[s].append(3)
                        # down
                        if lin_map[s+nrows]!= "H":
                            avail_actions[s].append(1)
                        
                # last colum
                elif (s+1)%nrows == 0:
                    if s == nrows-1:
                        # left
                        if lin_map[s-1]!= "H":
                            avail_actions[s].append(0)
                        
                        #down
                        if lin_map[s+nrows]!= "H":
                            avail_actions[s].append(1)
                        
                    elif s == (nrows**2)-1:
                        # left
                        if lin_map[s-1]!= "H":
                            avail_actions[s].append(0)
                        # up
                        if lin_map[s-nrows]!= "H":
                            avail_actions[s].append(3)
                
                    else:
                        # left
                        if lin_map[s-1]!= "H":
                            avail_actions[s].append(0)
                        # up
                        if lin_map[s-nrows]!= "H":
                            avail_actions[s].append(3)
                        # down
                        if lin_map[s+nrows]!= "H":
                            avail_actions[s].append(1)
                        
                # middle columns
                else:
                    if s < nrows:
                        # left
                        if lin_map[s-1]!= "H":
                            avail_actions[s].append(0)
                        # right
                        if lin_map[s+1]!= "H":
                            avail_actions[s].append(2)
                        # down
                        if lin_map[s+nrows]!= "H":
                            avail_actions[s].append(1)
                    elif s > nrows*(nrows-1):
                        # left
                        if lin_map[s-1]!= "H":
                            avail_actions[s].append(0)
                        # right
                        if lin_map[s+1]!= "H":
                            avail_actions[s].append(2)
                        # up
                        if lin_map[s-nrows]!= "H":
                            avail_actions[s].append(3)
                    else:
                        # right
                        if lin_map[s+1]!= "H":
                            avail_actions[s].append(2)
                        # down
                        if lin_map[s+nrows]!= "H":
                            avail_actions[s].append(1)
                        # left
                        if lin_map[s-1]!= "H":
                            avail_actions[s].append(0)
                        # up
                        if lin_map[s-nrows]!= "H":
                            avail_actions[s].append(3)
                            
        return avail_actions, holes
        
            
    def reset(self, init = False, state=None):
        """ reset the environment """
        raw_obs = self.env.reset()
        self.t = 0
        if self.use_action_masking:
            self.update_avail_actions(raw_obs)
            obs = {
                "action_mask": self.action_mask,
                "state": raw_obs,
            }
        else:
            obs = raw_obs
        
        return obs
    
    def render(self):
        self.env.render()
        

    def step(self, action):
        """ Adding some reward shaping """ 
        
        self.t += 1
        danger_tax = 0
        if self.use_run_time_assurance:
            probe_state, unsafe = self.probe_step(action)
            # switch to unsafe controller if unsafe
            if unsafe:
                danger_tax = -5
                action = self.get_safe_control()
                
        state, reward, done, info = self.env.step(action)
        
        # if self.testing == False:
        #     if done:
        #         reward += 75
        #     else:
        #         reward = -1 + danger_tax
        # Could make a custom reward here if you want
        if self.use_action_masking:
            self.update_avail_actions(state)
            obs = {
                "action_mask": self.action_mask,
                "state": state,
            }
        else:
            obs = state
            
        if self.t >= 100: # ------ change if using v1. Can make this to find shortest path
            done = True
        return obs, reward, done, info
    
    def probe_step(self, action):
        """ Probe step using environement dynamics. Checks if action will lead to unsafe state"""
        # how each action affects the state
        # update for multiple maps
        operator = {
            0: -1,
            1: 4,
            2: 1,
            3: -4
        }
        
        curr_state = self.env.s
        
        if curr_state == 15:
            return 15, 0
        
        next_state = curr_state + operator[action]
        unsafe = next_state in self.holes
        
        return next_state, unsafe
    
    def get_safe_control(self):
        """ Safe controller for RTA """ 
        return np.random.choice(self.avail_actions[self.env.s])
        

    def update_avail_actions(self, obs):
        """ currently thinking about position and velocity constraints. Might try position and theta constraints?"""
        self.action_mask = np.array([1.0] * self.action_space.n)
        
        # better way to do this, just first thing i thought of.
        
        for index in range(len(self.action_mask)):
            if index not in self.avail_actions[obs]:
                self.action_mask[index] = 0
                
                