"""
Reproduce Figure 4: 5 Item Knapsack SPGA-RTA vs SPGA-AM
"""

"""

Used to demo a trained agent in the Knapsack Environment environment --runtime.

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
import seaborn as sns
import sys
sys.path.append('/Users/probinet/Documents/PROJECTS/ICCPS_SPGA_REP/spga/action_masking/knapsack/utils')



# from run_time_assurance.knapsack.train_ga_knapsack_rta import expected_return
# from run_time_assurance.knapsack.train_ppo_knapsack_rta import get_ppo_trainer
# from run_time_assurance.knapsack.utils.ga_masking import Agent
# from run_time_assurance.knapsack.utils.custom_knapsack import Knapsack

from action_masking.knapsack.train_ga_knapsack_amask import expected_return
from action_masking.knapsack.train_ppo_knapsack_amask import get_ppo_trainer
from action_masking.knapsack.utils.ga_masking import Agent
from action_masking.knapsack.utils.custom_knapsack import Knapsack

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
        'mask': True,
        'runtime':False},
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
        'mask': True,
        'runtime': False},
    
    5: {'N': 5,
        'max_weight': 15,
        'item_weights': np.array([1, 12, 2, 1, 4]),
        'item_values': np.array([2, 4, 2, 1, 10]),
        'mask': True,
        'runtime': False}
}

OPTIMAL_SOLUTION = {
            5: 36,
            50: 104,
            100: 104,
}
            

def get_args():
    """
    Parse the command arguments
    """
    # create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments to be parsed
    parser.add_argument("--items", type=int, default=5, help="The map dimensions of the frozen lake")
    parser.add_argument("--seed", type=int, default=4, help="Indicate the training seed")
    parser.add_argument("--strategy", type=str, default="action_masking")
    args = parser.parse_args() 

    return args

def generate_plot(path, env_config, name):
    """
    path : dict
        path taken by an agent during a rollout
    env_config : config file
        environment configuration file
    name : str
        name to save the plot to
    
    """
    sns.set()
    csfont = {'fontname':'Times New Roman',  'fontsize':18}
    plt.figure(1)
    plt.plot(len(path["val"]), OPTIMAL_SOLUTION[env_config["N"]], 'o', markersize=18, color="#00FF00", label="Optimal Solution")
    plt.plot(range(1, len(path["val"]) + 1), path["val_sum"], '-o', color='green', label="Total Value")
    plt.plot(range(1, len(path["val"]) + 1), [env_config["max_weight"]]*len(path["val"]), '--', linewidth=3, color='red', label="Max Weight")
    plt.bar(range(1, len(path["lb_sum"]) + 1), path["lb_sum"], color='blue', label="Total Weight")
    plt.bar(range(1, len(path["lb"]) +1), path["lb"], color="orange", edgecolor="black", hatch="//", label="Chosen Item Weight")
    plt.legend(fontsize=12)
    plt.xticks(**csfont) 
    plt.yticks(**csfont)
    plt.ylabel("Value", **csfont)
    plt.xlabel("Step", **csfont)
    plt.tight_layout()
    plt.savefig(f"artifacts/knapsack_{name}_rta_{env_config['N']}_results.png", bbox_inches='tight', dpi=200)
    plt.show()
        
    return 0

def main():
    args = get_args()
    vers = int(args.items)
    assert (vers in [5, 50, 100]), "Item environment not available."
    #
    # Generate GA Path
    #
    ga_path = None
    ppo_path = None
    # GA
    env_config = ENV_DICT[vers]
    env_config["seed"] = args.seed
    agent = Agent()
    agent.load(f"action_masking/knapsack/trained_agents/knapsack_ga_masking_seed-{args.seed}_checkpoint-{args.items}.json")
    agent.strategy = args.strategy
    eval_rewards, ga_path = expected_return(agent, 1, env_config, agent_type='ga')
    #
    # Generate PPO Path
    #
    args.strategy = "action_masking"
    ppo_agent, ec = get_ppo_trainer(args)
    print("ECCC, env", ec, env_config)
    name = f"knapsack_ppo_masking_seed-{args.seed}_checkpoint-{args.items}"
    ppo_agent.restore("action_masking/knapsack/trained_agents/{}/{}".format(name, name))
    eval_rewards, ppo_path = expected_return(ppo_agent, 1, env_config)

    generate_plot(ga_path, env_config, "SPGA")
    generate_plot(ppo_path, env_config, "SRL")
    
    return 0
    
    
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("---------------")
    print("Total time: ", abs(end-start))