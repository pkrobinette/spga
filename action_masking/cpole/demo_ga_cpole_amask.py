from ray import tune
import json
import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle

from train_ga_cpole_amask import final_evaluation
from utils.ga_masking import Agent

def get_args():
    """
    Parse the command arguments
    """
    # create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments to be parsed
    parser.add_argument("--num_rollouts", type=int, default=1, help="Number of times to rollout agent in env")
    parser.add_argument("--render", choices=('True','False'), help="Render the rollout")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--x_thresh", type=float, default=1.5)
    args = parser.parse_args()
    args.render = True if args.render == 'True' else False

    return args

def main():
    args = get_args()
    #
    # Demo With Action Asking
    #
    env_config = {"use_action_masking": True}
    agent = Agent()
    agent.load("trained_agents/seed_{}/cartpole_ga_masking_seed-{}_checkpoint-{}.json".format(args.seed, args.seed, str(args.x_thresh)))
    agent.strategy = "action_masking"
    mask_eval_reward, mask_eval_time, mask_v_total, mask_v_eps = final_evaluation(agent, args.num_rollouts, env_config)
    
    print("\n----- Demo With Masking -----")
    print("Avg. Rollout Reward WITH Masking: ", mask_eval_reward)
    print("Avg. Rollout Time WITH Masking: ", mask_eval_time)
    print("Total Violations WITH Masking: ", mask_v_total)
    print("Percentage of Safe Rollouts WITH Masking: {}%".format(100-(mask_v_eps/args.num_rollouts*100)))
    
    
if __name__ == "__main__":
    main()