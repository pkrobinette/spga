"""
Genetic Algorithm with action masking and manipulated reward.

"""


import numpy as np
# np.random.seed(1)           # set random seed for repeatability
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import copy
import time
import tensorflow as tf


def single_relu(x):
    if x < 0:
        return 0
    else:
        return x

relu = np.vectorize(single_relu)

class Agent:

    def __init__(self, input_size = 4, hidden_size = 32, output_size = 2, std = 0.5, mute_rate = 0.2, strategy=None,continuous=False):
        self.std = std                  # standard deviation
        self.reward = 0                 # Cumulative total reward during an episode
        self.mute_rate = mute_rate      # Mutation rate
        self.cont = continuous
        self.strategy=strategy
        
        # TODO: maybe the biases shouldn't all be initalized to .1??
        self.b1 = np.ones(hidden_size,)         
        self.w1 = np.random.randn(input_size, hidden_size)

        self.b2 = np.random.randn(output_size,)
        self.w2 = np.random.randn(hidden_size, output_size)


    def get_action(self, state):
        """
        Generate an action from a given state (or observation). Current implementation uses relu activation function.
        """
        if self.strategy == "action_masking":
             # Extract the available actions tensor from the observation.
            action_mask = state["action_mask"]

            # Compute the unmasked logits.
            reshape_state = np.hstack(state["state"])
            l1 = relu(reshape_state.dot(self.w1) + self.b1)
            logits = l1.dot(self.w2) + self.b2
            
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = tf.maximum(tf.math.log(action_mask.astype(float)), tf.float32.min)
            
            # Return masked logits.
            return np.argmax(logits + np.array(inf_mask))
        else:
#             if state.shape[0] == 3:
#                 state = np.hstack((state[0], state[1], state[2]))
#             elif state.shape[0] == 2:
#                 state = np.hstack((state[0], state[1]))
#             l1 = relu(state.dot(self.w1) + self.b1)
#             output = l1.dot(self.w2) + self.b2
            
#             if self.cont:
#                 return np.tanh(output)
    
#             # check that this returns either -1 or +1
#             return np.argmax(output)
            if state.shape[0] == 3:
                state = np.hstack((state[0], state[1], state[2]))
            elif state.shape[0] == 2:
                state = np.hstack((state[0], state[1]))
            # reshape_state = np.hstack(state["state"])
            l1 = relu(state.dot(self.w1) + self.b1)
            output = l1.dot(self.w2) + self.b2
            
            # return output[0]
        
            if self.cont:
                return np.tanh(output)
            
    
            # check that this returns either -1 or +1
            return np.argmax(output)
        


    def mutate(self):
        """
        Mutate weights and biases for an offspring of selected agents.

        Returns:
        -------
        New weights and biases.
        """
        # TODO: find in_l, hl, and out_l at one time instead of continually calling the function

        # generate array of size inxhl
        rnd_w1 = np.random.uniform(0, 1, [self.w1.shape[0], self.w1.shape[1]])
        rnd_w2 = np.random.uniform(0, 1, [self.w2.shape[0], self.w2.shape[1]])

        # TODO: validate changes to stay within valid number range?
        chck_rnd1 = np.where(rnd_w1 < self.mute_rate, np.random.uniform(-self.std, self.std, [self.w1.shape[0], self.w1.shape[1]]), 0)
        chck_rnd2 = np.where(rnd_w2 < self.mute_rate, np.random.uniform(-self.std, self.std, [self.w2.shape[0], self.w2.shape[1]]), 0)

        # updates weights and biases
        self.w1 += chck_rnd1
        self.w2 += chck_rnd2

        self.b1 += np.random.uniform(0, self.std, self.w1.shape[1])
        self.b2 += np.random.uniform(0, self.std, self.w2.shape[1])


    def update_reward(self, reward):
        """ Update the cumulative sum of rewards with feedback from the environment """
        self.reward += reward

    def reset(self):
        """ Reset the cumulative sum of rewards """
        self.reward = 0

    def save(self, name):
        """ Save agent information to a json file """
        output_dict = {
            'w1': list(self.w1.tolist()),
            'b1': list(self.b1.tolist()),
            'w2': list(self.w2.tolist()),
            'b2': list(self.b2.tolist())
        }
        with open(name+'.json', 'w') as fp:
            json.dump(output_dict, fp)

    def load(self, name):
        """ Load an agent from a json file """
        f = open(name)
        info = json.load(f)
        self.w1 = info['w1']
        self.b1 = info['b1']
        self.w2 = info['w2']
        self.b2 = info['b2']
        f.close()



class Generation:

    def __init__(self, env, continuous=False, n=20, std=.1, g=100, solved_score=190, e_rate=0.2, mute_rate=0.2, hidd_l=32, strategy=None, seed=None):
        assert (env != None), "Must include an environment"
        # Parameters
        self.generations = g      # number of generations
        self.elite_rate = e_rate  # elitism rate
        self.std = std            # standard deviation
        self.n = n                # number of individuals in a population
        self.best_fitness = []      # record best fitness across generations
        self.avg_fitness = []       # record average fitness across generations
        self.avg_fitness_history = []   # record fitness history across generations
        self.solved_score = solved_score
        self.set_elite_num()
        self.mute_rate = mute_rate
        self.cont = continuous
        self.running_total = 0
        self.strategy = strategy
        # if strategy == "action_masking":
        #     print("Using X Threshold: {}\n".format(env.x_threshold))
        
        if seed:
            np.random.seed(seed)

        # Environment Info
        self.env = env
        if continuous:
            self.action_dim = env.action_space.shape[0]
        else:
            self.action_dim = env.action_space.n
            
        # if strategy == "action_masking":
        #     x_dim, y_dim = env.observation_space["state"].shape
        #     self.state_dim = x_dim * y_dim
        #     # self.state_dim = env.observation_space["state"].shape[0]
        # else:
        #     x_dim, y_dim = env.observation_space.shape
        #     self.state_dim = x_dim * y_dim
        self.state_dim = None
        if strategy == "action_masking":
            self.state_dim = env.observation_space["state"].shape[0]
        else:
            x_dim, y_dim = env.observation_space.shape
            self.state_dim = x_dim * y_dim
            
        self.hidd_l = hidd_l

        # Population
        self.agents = [Agent(input_size=self.state_dim, hidden_size=hidd_l, output_size=self.action_dim, 
                             std=std, mute_rate=mute_rate, strategy=self.strategy,continuous=self.cont) for _ in range(n)]
        self.ag_rewards = np.empty(self.n)  # stores rewards for each individual in this generation
        self.best_agent = Agent(input_size=self.state_dim, hidden_size=hidd_l, output_size=self.action_dim, 
                                std=std, mute_rate=mute_rate, strategy=self.strategy, continuous=self.cont)
        self.solved_score = solved_score
        
        
    def set_elite_num(self):
        """ 
        Set the number of elite individuals for each generation
        """
        self.elite_num = int(round(self.n * self.elite_rate))
        
        if self.elite_num%2 !=0:
            self.elite_num += 1


    def reset(self):
        """ 
        Reset reward values for each agent in the population 
        """
        for agent in self.agents:
            agent.reset()
        self.ag_rewards = np.empty(self.n)


    def crossover(self, p1, p2):
        """
        Crossover and two selected parents
        """
        in_l = p1.w1.shape[0]
        hl = p1.w1.shape[1]
        out_l =p1.w2.shape[1]
            
        # # Two Point Crossover
        rnd_1 = np.random.randint(hl+1)
        rnd_2 = np.random.randint(rnd_1, hl+1)
        
        w1 = np.hstack((p1.w1[:, :rnd_1], p2.w1[:, rnd_1:rnd_2], p1.w1[:, rnd_2:]))
        w2 = np.hstack((p1.w2[:, :rnd_1], p2.w2[:, rnd_1:rnd_2], p1.w2[:, rnd_2:]))
        
        b1 = np.concatenate((p1.b1[:rnd_1], p2.b1[rnd_1:rnd_2], p1.b1[rnd_2:]))
        b2 = np.concatenate((p1.b2[:rnd_1], p2.b2[rnd_1:]))

        agent = Agent(input_size=self.state_dim, hidden_size=self.hidd_l, output_size=self.action_dim, 
                      std=self.std, mute_rate=self.mute_rate, strategy=self.strategy, continuous=self.cont)
        agent.w1 = w1
        agent.b1 = b1
        agent.w2 = w2
        agent.b2 = b2
            
        return agent
            
    
    def TournamentSelect(self):
        """ 
        Selection of agents using the tournament selection 
        """
        new_gen = []
        #
        # elitism
        #
        sorted_rewards = np.argsort(-self.ag_rewards)
        for i in range(self.elite_num):
            new_gen.append(self.agents[sorted_rewards[i]])
        #
        # tournament selection
        #   
        while len(new_gen) < self.n:
            i1, i2, i3, i4 = np.random.randint(0, self.n, 4)
            if self.agents[i1].reward > self.agents[i2].reward:
                p1 = self.agents[i1]
            else:
                p1 = self.agents[i2]
                
            if self.agents[i3].reward > self.agents[i4].reward:
                p2 = self.agents[i3]
            else:
                p2 = self.agents[i4]
            
            new_gen.append(self.crossover(p1, p2))
            
        return new_gen



    def calc_fitness(self):
        """ 
        Calculate fitness by rolling out each agent in the cartpole environment.
        Fitness is the cumulative total reward during the rollout.
        """
        for i in range(self.n):
            agent = self.agents[i]
            obs = self.env.reset()
            while True:
                action = agent.get_action(obs)
                obs, reward, done, _ = self.env.step(action)
                agent.update_reward(reward)
                if done:
                    self.ag_rewards[i] = agent.reward
                    break


    def simulate(self):
        """ 
        Simulate the evolution of the population for n generataions 
        
        Returns
        -------
        Best NN after g generations
        """
        for k in range(self.generations):
            #
            # Reset each agent's reward value
            #
            self.reset()
            #
            # Calculate the fitness score for each agent in the population
            #
            self.calc_fitness()
          
            new_gen = self.TournamentSelect()  # *comment out if using RouletteWheel
  
            for agent in new_gen:
                agent.mutate()
            #
            # Record population stats
            #
            self.best_fitness.append(max(self.ag_rewards))
            self.avg_fitness.append(np.mean(self.ag_rewards))
            self.avg_fitness_history.append(np.mean(self.best_fitness[-30:]))
            #
            # Print best fitness value every 50 generations
            #
            if k%10 == 0 or k==self.generations-1:
                print(f"Generation number:{k} | Best Fitness Value: {self.best_fitness[-1]} | Average Fitness: {self.avg_fitness[-1]}")
            #
            # Stopping Criteria
            #
            if self.avg_fitness_history[-1] >= self.solved_score and k > 30:
                print('Solved score threshold reached. Stopping training ...')
                break
            #
            # Update new generation
            #
            start = time.time()
            self.agents = new_gen
            end = time.time()
            
        # Calculate final reward values of last generation and set best agent
        self.reset()
        self.calc_fitness()
        # self.fitness_history.append(self.ag_rewards)
        self.best_fitness.append(max(self.ag_rewards))
        self.avg_fitness.append(np.mean(self.ag_rewards))
        self.avg_fitness_history.append(np.mean(self.best_fitness[-30:]))
        index = np.argmax(self.ag_rewards)
        self.best_agent = copy.deepcopy(self.agents[index])