"""
RTA CartPole environment from OpenAI Gym adjusted for action masking.

Resources:
-----------
https://github.com/PacktPublishing/Mastering-Reinforcement-Learning-with-Python/blob/master/Chapter10/custom_mcar.py

"""


import gym
from gym.spaces import Box, Dict
import numpy as np
import math


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
        self.t = 0 # steps
        self.x_threshold = env_config.get("x_threshold", 1.5) # boundary
        self.use_action_masking = env_config.get("use_action_masking", False)
        self.use_run_time_assurance = env_config.get("use_run_time_assurance", False)
        # extend the initial start conditions. Not used in this work.
        self.extend = env_config.get("extend", False)
        self.action_mask = None
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
        Reset the environment
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
        if self.use_run_time_assurance:
            probe_state, unsafe = self.probe_step(action)
            # switch to safe controller if unsafe
            if unsafe:
                x, x_dot, theta, theta_dot = probe_state
                # go right
                if x <= -self.x_threshold: # go right
                    action = 1
                elif x>= self.x_threshold: # go left
                    action = 0 
                
        state, reward, done, info = self.env.step(action)
        # Could make a custom reward here if you want
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
    
    
    def probe_step(self, action):
        """
        Probe step using environment dynamics. Checks if the proposed action will lead
        to an unsafe state.
        """
        x, x_dot, theta, theta_dot = self.env.state
        force = self.env.force_mag if action == 1 else -self.env.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
            force + self.env.polemass_length * theta_dot ** 2 * sintheta
        ) / self.env.total_mass
        thetaacc = (self.env.gravity * sintheta - costheta * temp) / (
            self.env.length * (4.0 / 3.0 - self.env.masspole * costheta ** 2 / self.env.total_mass)
        )
        xacc = temp - self.env.polemass_length * thetaacc * costheta / self.env.total_mass

        if self.env.kinematics_integrator == "euler":
            x = x + self.env.tau * x_dot
            x_dot = x_dot + self.env.tau * xacc
            theta = theta + self.env.tau * theta_dot
            theta_dot = theta_dot + self.env.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.env.tau * xacc
            x = x + self.env.tau * x_dot
            theta_dot = theta_dot + self.env.tau * thetaacc
            theta = theta + self.env.tau * theta_dot

        probe_state = (x, x_dot, theta, theta_dot)

        unsafe = bool(
            x <= -self.x_threshold
            or x >= self.x_threshold
        #     or theta <= -THETA_THRESHOLD
        #     or theta >= THETA_THRESHOLD
        )

        return np.array(probe_state, dtype=np.float32), unsafe

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