from utils.masking_model import ActionMaskModel, ActionMaskModel50, ActionMaskModel100
from utils.custom_knapsack import Knapsack

import ray
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from ray.tune.utils.util import SafeFallbackEncoder
import ray.rllib.agents.ppo as ppo
from ray import tune
import json
import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
from train_ppo_knapsack_rta import get_ppo_trainer


import sys
sys.path.append(sys.path[0]+"/results")
sys.path.append(sys.path[0]+"/trained_agents")
sys.path.append(sys.path[0]+"/utils")

def get_args():
    """
    Parse the command arguments
    """
    # create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments to be parsed
    parser.add_argument("--num_rollouts", type=int, default=1, help="Number of times to rollout agent in env")
    parser.add_argument("--render", choices=('True','False'), help="Render the rollout")
    parser.add_argument("--strategy", default="runtime")
    parser.add_argument("--items", type=int, default=5, help="The items in the Knapsack")
    parser.add_argument("--seed", type=int, default=4, help="Indicate the training seed")
    args = parser.parse_args()
    args.render = True if args.render == 'True' else False

    return args

def final_evaluation(trainer, n_final_eval, env_config={}):
    """
    Used for final evaluation policy rollout.
    
    Parameters:
    -----------
    trainer : ppo agent
    n_final_eval : int
        number of times to evaluate an agent
    env_config : dict
        environment configuration file
        
    Returns
    --------
    - mean of all eval rewards
    - mean of all rollout times
    - number of total violations
    - number of episodes with at least one violation
    - path
    """
    action_masking = env_config.get("mask", False)
    env = Knapsack(env_config)
    eval_rewards = []
    eval_time = []
    v_total = 0
    v_eps = 0
    
    for _ in range(n_final_eval):
        obs = env.reset()
        r = 0
        steps = 0
        safe = True
        path = {"lb":[], "val":[], "lb_sum":[], "val_sum":[]}
        while True:
            if env.current_weight > env.max_weight:
                if safe:
                    v_eps += 1
                    safe = False
                v_total += 1
            action = trainer.compute_single_action(obs)
            # print("Step {} --> Weight: {} | Value: {}".format(steps, env.item_weights[action], env.item_values[action]))
            path["lb_sum"].append(sum(path["lb"]) + env.item_weights[action])
            path["val_sum"].append(sum(path["val"]) + env.item_values[action])
            path["lb"].append(env.item_weights[action])
            path["val"].append(env.item_values[action])
            obs, reward, done, _ = env.step(action)
            r += reward
            steps += 1
            if done:
                eval_rewards.append(r)
                eval_time.append(steps)
                break
    
    return np.mean(eval_rewards), np.mean(eval_time), v_total, v_eps, path


def main():
    """
    main function
    """
    args = get_args()
    #
    # Demo With Action Asking
    #
    mask_agent, mask_env_config = get_ppo_trainer(args)
    name = "knapsack_ppo_rta_seed-{}_checkpoint-{}".format(args.seed, str(args.items))
    mask_agent.restore("trained_agents/{}/{}".format(name, name))
    mask_eval_reward, mask_eval_time, mask_v_total, mask_v_eps, path = final_evaluation(mask_agent, args.num_rollouts, env_config=mask_env_config)
    
    
    print("\n----- Demo -----")
    print("Avg. Rollout Reward: ", mask_eval_reward)
    print("Avg. Num of Steps: ", mask_eval_time)
    print("Total Violations: ", mask_v_total)
    print("Percentage of Safe Rollouts: {}%".format(100-(mask_v_eps/args.num_rollouts*100)))
    
    
    
if __name__ == "__main__":
    main()