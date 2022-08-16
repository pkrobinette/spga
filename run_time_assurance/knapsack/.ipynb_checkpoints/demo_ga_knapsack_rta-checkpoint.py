"""

Used to demo a trained agent in the FrozenLake environment.

To RUN:
1. Set env_config in main() to reflect the size of the map (e.g, 8x8, 16x16,..)
2. set the trained agent in the `agent.load` function

"""

from ray import tune
import json
import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle

from train_ga_knapsack_rta import final_evaluation
from utils.ga_masking import Agent
from utils.custom_knapsack import Knapsack

def get_args():
    """
    Parse the command arguments
    """
    # create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments to be parsed
    parser.add_argument("--num_rollouts", type=int, default=1, help="Number of times to rollout agent in env")
    parser.add_argument("--render", choices=('True','False'), help="Render the rollout")
    parser.add_argument("--items", type=int, default="5", help="The number of items in the knapsack")
    parser.add_argument("--seed", type=int, default=4, help="Indicate the training seed")
    args = parser.parse_args()
    args.render = True if args.render == 'True' else False

    return args

def final_evaluation(agent, num_rollouts, env_config={}):
    """
    Used for final evaluation policy rollout
    """
    action_masking = env_config.get("mask", False)
    env = Knapsack(env_config)
    eval_rewards = []
    eval_time = []
    v_total = 0
    v_eps = 0
    
    for _ in range(num_rollouts):
        safe = True
        steps = 0
        r = 0
        obs = env.reset()
        path = {"lb":[], "val":[], "lb_sum":[], "val_sum":[]}
        while True:
            if env.current_weight > env.max_weight:
                if safe:
                    v_eps += 1
                    safe = False
                v_total += 1
            
            action = agent.get_action(obs)
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
    args = get_args()
    # assert seed is a valid entry
    assert (args.seed in [98, 4, 36, 27, 2]),"Not a valid training seed"
    
    #
    # Demo With Action Asking
    #
    # env_config = {"use_action_masking": True, "map_name":args.map}
    # env = FrozenLake(env_config=env_config)
    if args.items == 50:
        env_config = {
            'N': 50,
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
            'mask': False,
            'runtime':True,
            'seed': args.seed}
            
    elif args.items == 100:
        env_config = {
            'N': 100,
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
            'mask': False,
            'runtime':True,
            'seed': args.seed}
            
    else:
        env_config = {
            'N': 5,
            'max_weight': 15,
            'item_weights': np.array([1, 12, 2, 1, 4]),
            'item_values': np.array([2, 4, 2, 1, 10]),
            'mask': False,
            'runtime':True,
            'seed': args.seed}
        
    agent = Agent()
    agent.load("trained_agents/knapsack_ga_rta_seed-{}_checkpoint-{}.json".format(args.seed, str(args.items)))
    agent.strategy = None
    # agent.strategy = "action_masking"
    mask_eval_reward, mask_eval_time, mask_v_total, mask_v_eps, path = final_evaluation(agent, args.num_rollouts, env_config)
    
    print("\n----- Demo With RTA -----")
    print("Avg. num of steps to goal: ", mask_eval_time)
    print("Avg. Rollout Reward WITH RTA: ", mask_eval_reward)
    print("Total Violations WITH RTA: ", mask_v_total)
    print("Percentage of Safe Rollouts WITH RTA: {}%".format(100-(mask_v_eps/args.num_rollouts*100)))
    
    
    
    
if __name__ == "__main__":
    main()