"""
Run rollouts of trained agents to determine their expected return.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join
import json
# from utils.custom_rta_cpole import CartPole
from train_ppo_cpole_rta import get_ppo_trainer
from utils.ga_masking import Agent
from matplotlib.patches import Rectangle
from matplotlib import colors as mcolors
from scipy.signal import savgol_filter
import argparse
from utils.utils import rollout
import warnings
warnings.filterwarnings("ignore")
import pickle
import time
import pandas as pd


NUM_TRIALS = 500 # number of rollouts to run
SEED = 4 # which training seed to evaluate
np.random.seed(10)

def get_args():
    """ parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--strategy', type=str, default='masking', help='masking or rta')
    args = parser.parse_args()
    
    return args

def main(args):
    #
    # Data Organization
    #
    exp = {
        1.5:{"ga":[], "ppo":[]},
        1.25:{"ga":[], "ppo":[]},
        1.0:{"ga":[], "ppo":[]},
        0.75:{"ga":[], "ppo":[]},
        0.5:{"ga":[], "ppo":[]},
        0.25:{"ga":[], "ppo":[]},
    }
    #
    # initialize testing environment vars.
    #
    init_pts = np.random.uniform(low=-0.05, high=0.05, size=(NUM_TRIALS,4))      
    env_config = {"use_action_masking": False}
    #
    # Run rollouts for each version of the environment
    #
    for key in exp.keys():
        print(f"Running Vers.: {key}...")
        #
        # Test a GA agent
        #
        ga_agent = Agent()
        ga_agent.load(f"trained_agents/seed_{SEED}/cartpole_ga_{args.strategy}_seed-{SEED}_checkpoint-{key}.json")
        ga_agent.strategy = None
        ga_exp_return, _, _, _, _, _ = rollout(ga_agent, init_pts, env_config, ag_type='ga')
        exp[key]['ga'] = ga_exp_return
        #
        # Test a SRL Agent
        #
        ppo_agent, ec = get_ppo_trainer()
        name = f"cartpole_ppo_{args.strategy}_seed-{SEED}_checkpoint-{key}"
        ppo_agent.restore(f"trained_agents/seed_{SEED}/{name}/{name}")
        ppo_exp_return, _, _, _, _, _ = rollout(ppo_agent, init_pts, env_config)
        exp[key]['ppo'] = ppo_exp_return
        print("GA: ", sum(ga_exp_return)/NUM_TRIALS)
        print("PPO: ", sum(ppo_exp_return)/NUM_TRIALS)
        print('\n')
    #
    # SAVE expected return values
    #
    print('Saving data ...')
    for vers in exp.keys():
        for mode in exp[vers].keys():
            save_name = f"results/expected_return/cpole_{mode}_{args.strategy}_seeded_expreturn-{vers}.pkl"
            with open(save_name, 'wb') as f:
                pickle.dump(exp[vers][mode], f)
    
    df = pd.DataFrame(exp)
    df.to_csv('results/expected_return/combined.csv')
        
    return 0
    
if __name__ == "__main__":
    #
    # Get Args
    #
    args = get_args()
    #
    # Run main
    #
    start = time.time()
    main(args)
    end = time.time()
    
    print("------------------------")
    print("Total time: ", abs(end-start))


