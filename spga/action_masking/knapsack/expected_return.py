"""

Used to demo a trained agent in the Knapsack Environment environment.

"""

from ray import tune
import json
import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import time

from train_ga_knapsack_amask import expected_return
from train_ppo_knapsack_amask import get_ppo_trainer
from utils.ga_masking import Agent
from utils.custom_knapsack import Knapsack

ENV_DICT = {
    50: {'N': 50,
        'max_weight': 75,
        'item_weights': np.array([40,  7, 32, 26, 13, 31, 43, 13, 11, 28, 
                                  9, 48, 42, 28, 45, 17, 45, 46, 26,  6, 22, 
                                  8, 18, 29, 42, 49, 27, 14, 46, 34, 33, 32, 
                                  18, 14, 14, 48, 32, 24, 40, 22, 19, 37, 37,
                                  18, 37, 29, 11, 35, 37, 11]),
        'item_values': np.array([17, 10,  7, 11, 10,  9,  2, 15,  8,  8,  7,
                                 12,    9, 12,  4, 11, 18, 17,  5,  5,  4,  
                                 3, 10, 13,  9, 14, 12, 11,  1, 14, 13,  6, 
                                 16,  8, 13, 14, 19, 18, 17,  3, 16,  3, 13,  
                                 2, 10, 16, 14,  4, 11, 14]),
        'mask': True},
    100: {'N': 100,
        'max_weight': 50,
        'item_weights': np.array([33, 21, 29, 17, 33, 32, 35, 23, 22, 
                                  18, 49, 47,  8, 41, 49, 45, 29, 49, 43, 49, 15,  
                                  7, 46, 47, 18, 46, 20, 38, 38, 30, 37, 39, 30, 48, 
                                  35, 19, 16, 31, 26, 41, 21,  7, 48, 41, 28, 37, 11, 
                                  6,  9, 33, 30, 12, 14, 25, 40, 22, 33, 33, 24, 16, 
                                  15, 12,  6, 49, 33, 35, 33, 39, 5, 34,  6, 43, 48, 
                                  25, 33, 11, 31, 46, 26, 26, 20, 47, 46, 46, 46, 14,
                                  37, 27, 40, 12, 21, 45, 21, 21, 15, 38, 10, 37, 21, 27]),
        'item_values': np.array([ 6, 14,  6, 14,  7,  5, 14,  8, 14, 11, 12, 12, 
                                 12, 14, 10,  2, 10,3,  6,  1,  3,  2,  8, 10,  2, 
                                 12,  4,  2,  4,  8, 14,  3,  6,  8, 4,  2, 14,  8,  
                                 7, 13,  5,  2,  5, 12,  2,  1, 11, 11,  2,  8,  5, 
                                 9,  5,  2, 13,  9, 12,  7,  7,  8,  7, 11,  4,  2,  
                                 4,  5,  5, 13, 4,  6, 13,  8,  5,  4,  2, 11, 11, 
                                 10, 13,  7,  7,  5, 14,  3,  2, 14, 14,  5, 11, 12, 
                                 12,  6,  6, 12,  5,  2, 10,  5,  8, 14]),
        'mask': True},
    5: {'N': 5,
        'max_weight': 15,
        'item_weights': np.array([1, 12, 2, 1, 4]),
        'item_values': np.array([2, 4, 2, 1, 10]),
        'mask': True}
}
            

def get_args():
    """
    Parse the command arguments
    """
    # create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments to be parsed
    parser.add_argument("--num_rollouts", type=int, default=500, help="Number of times to rollout agent in env")
    parser.add_argument("--render", choices=('True','False'), help="Render the rollout")
    parser.add_argument("--items", type=str, default="5", help="The map dimensions of the frozen lake")
    parser.add_argument("--seed", type=int, default=4, help="Indicate the training seed")
    parser.add_argument("--strategy", default="action_masking")
    parser.add_argument
    args = parser.parse_args()
    args.render = True if args.render == 'True' else False

    return args

def main():
    args = get_args()
    #
    # Data Organization
    #
    exp = {
        5:{"ga":[], "ppo":[]},
        50:{"ga":[], "ppo":[]},
        100:{"ga":[], "ppo":[]},
    }   
    #
    # Expected Return for all GA Agents
    #
    ga_path = None
    ppo_path = None
    method = "masking"
    # GA
    print("------GA KNAPSACK-------")
    for vers in exp.keys():
        args.items = vers
        env_config = ENV_DICT[vers]
        env_config["seed"] = args.seed
        agent = Agent()
        agent.load(f"trained_agents/knapsack_ga_{method}_seed-{args.seed}_checkpoint-{vers}.json")
        agent.strategy = args.strategy
        eval_rewards, ga_path = expected_return(agent, args.num_rollouts, env_config, agent_type='ga')
        exp[vers]['ga'] = eval_rewards
        print(f"Vers: {vers} || MEAN: {sum(eval_rewards)/args.num_rollouts}")
    #
    # Expected return for all PPO Agents
    #
    print("------PPO KNAPSACK-------")
    for vers in exp.keys():
        args.items = vers
        env_config = ENV_DICT[vers]
        env_config["seed"] = args.seed
        ppo_agent, _ = get_ppo_trainer(args)
        name = f"knapsack_ppo_{method}_seed-{args.seed}_checkpoint-{vers}"
        ppo_agent.restore("trained_agents/{}/{}".format(name, name))
        eval_rewards, ppo_path = expected_return(ppo_agent, args.num_rollouts, env_config)
        exp[vers]['ppo'] = eval_rewards
        print(f"Vers: {vers} || MEAN: {sum(eval_rewards)/args.num_rollouts}")
    #
    # Save Data
    #
    print('Saving data ...')
    for vers in exp.keys():
        for mode in exp[vers].keys():
            save_name = f"results/expected_return/knapsack_{mode}_{method}_seeded_expreturn-{vers}.pkl"
            with open(save_name, 'wb') as f:
                pickle.dump(exp[vers][mode], f)
                
    df = pd.DataFrame(exp)
    df.to_csv('results/expected_return/combined.csv')
        
    return 0
    
    
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("---------------")
    print("Total time: ", abs(end-start))