"""
Train agent with PPO using action masking in the CartPole-v0 environment.

Tuned Hyperparameters From : https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/cartpole-ppo.yaml
"""

from utils.masking_model import ActionMaskModel, ActionMaskModel50, ActionMaskModel100
from utils.custom_knapsack import Knapsack

import ray
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from ray.tune.utils.util import SafeFallbackEncoder
import ray.rllib.agents.ppo as ppo
from ray.rllib.utils.filter import MeanStdFilter
from ray import tune
import json
import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
import os
from os.path import isdir, join, isfile
import shutil

import or_gym
from or_gym.utils import create_env

# time to train, average of 100 rollouts, avg. episode length of 100 rollouts



def get_args():
    """
    Parse the command arguments
    """
    # create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments to be parsed
    parser.add_argument("--num_trials", type=int, default=1, help="Number of times to repeat training")
    parser.add_argument("--stop_reward", type=float, default=35.5, help="Stopping reward criteria for training")
    parser.add_argument("--env_name", type=str, default="knapsack", help="Name of the environment")
    parser.add_argument("--strategy", type=str, default='runtime', help="Training strategy")
    parser.add_argument("--num_eval_eps", type=int, default=20, help="Number of episodes to evaluate the trained agent on after training")
    parser.add_argument("--max_steps", type=int, default=500, help="Max number of training steps during the training process")
    parser.add_argument("--seed", type=int, default=12, help="Training seed to set randomization for training")
    parser.add_argument("--items", type=int, default=5, help="Number of items to train with")
    args = parser.parse_args()

    return args


def register_env(env_name, env_config={}):
    env = create_env(env_name)
    tune.register_env(env_name, 
        lambda env_name: env(env_name,
            env_config=env_config))
    

def min_print(result):
    """ Print results for each training step """
    result = result.copy()
    info_keys = [
        'episode_len_mean', 
        'episode_reward_max', 
        'episode_reward_mean',
        'episode_reward_min'
            ]
    out = {}
    for k, v in result.items():
        if k in info_keys:
            out[k] = v

    cleaned = json.dumps(out, cls=SafeFallbackEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)


def get_ppo_trainer(args= None):
    """ Configure the ppo trainer based on training strategy """
    env_name = "Knapsack-v0" # will need to change if using Knapsack-v1
    env_config = {
                'N': 5,
                'max_weight': 15,
                'item_weights': np.array([1, 12, 2, 1, 4]),
                'item_values': np.array([2, 4, 2, 1, 10]),
                'mask': True}
    
    register_env(env_name, env_config) # HAVE TO REGISTER KNAPSACK ENV
    config = ppo.DEFAULT_CONFIG.copy()
    config["env"] = Knapsack
    config["env_config"] = env_config
    
    if args:
        # add stuff for different size maps
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
                'mask': True}
            config["env_config"].update(env_config)
            
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
                'mask': True}
            config["env_config"].update(env_config)
            
        if args.strategy == "action_masking":
            print('\nUsing action masking to train ...\n')
            if args.items == 5:
                ModelCatalog.register_custom_model("kp_mask", ActionMaskModel)
                config["model"] = {
                    "custom_model": "kp_mask",
                }
                
            elif args.items == 50:
                ModelCatalog.register_custom_model("kp_mask", ActionMaskModel50)
                config["model"] = {
                    "custom_model": "kp_mask",
                }
                
            elif args.items == 100:
                ModelCatalog.register_custom_model("kp_mask", ActionMaskModel100)
                config["model"] = {
                    "custom_model": "kp_mask",
                }
                
            config["env_config"].update({"mask":True})
        elif args.strategy == "runtime":
            config["env_config"].update({"runtime":True})
            
        config["env_config"]["seed"] = args.seed
        config.update({'seed': args.seed})
        
    trainer = ppo.PPOTrainer(config=config)
    return trainer, config["env_config"]


def final_evaluation(trainer, n_final_eval, env_config={}):
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
            print("Step {} --> Weight: {} | Value: {}".format(steps, env.item_weights[action], env.item_values[action]))
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
    #
    # Setup and seeds
    #
    args = get_args()
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    print("STOP REWARD: ", args.stop_reward)

    # np.random.seed(args.seed)
    #
    # Train
    #
    # to be able to access checkpoint after training
    checkpoint = None
    train_time = []
    trainer = None
    ep_reward = None
    avg_ep_reward = None
    rand_seeds = [4, 36, 27, 2, 98]
    # rand_seeds = [44]
    env_config = None
    for i in range(args.num_trials):
        args.seed = rand_seeds[i]
        trainer, env_config = get_ppo_trainer(args)
        results = None
        training_steps = 0
        # Training
        ep_reward = []
        avg_ep_reward = []
        print('\n')
        while True:
            results = trainer.train()
            if (training_steps)%10 == 0:
                print("Training Step {} Results --------------".format(training_steps))
                print(min_print(results))
            ep_reward.append(results["episode_reward_mean"])
            avg_ep_reward.append(np.mean(ep_reward[-30:]))
            training_steps += 1
            if (avg_ep_reward[-1] >= args.stop_reward and training_steps >=30) or training_steps > args.max_steps:
                break
        #
        # save the trained agent
        #
        name = "knapsack_ppo_rta_seed-{}_checkpoint-{}".format(str(args.seed), str(args.items))
        train_time.append(results["time_total_s"])
        checkpoint= trainer.save("./trained_agents/"+name)
        for f in os.listdir("./trained_agents/"+name):
            ckpt_dir = join("./trained_agents/"+name, f)
            if not f.startswith('.') and isdir(ckpt_dir):
                for p in os.listdir(ckpt_dir):
                    test = join(ckpt_dir, p)
                    if not p.startswith('.'):
                        if "tune_metadata" in p:
                            os.rename(test, "./trained_agents/"+name+"/"+name+".tune_metadata")
                        else:
                            os.rename(test, "./trained_agents/"+name+"/"+name)
                shutil.rmtree(ckpt_dir)
                        
        checkpoint = "trained_agents/"+name+"/"+name
    #
    # Evaluate with Action Masking
    #
    # agent, env_config = get_ppo_trainer()
    # agent.restore(checkpoint)
    print("Training Complete. Testing the trained agent with action masking ...\n")
    
    mask_eval_reward, mask_eval_time, mask_v_total, mask_v_eps = final_evaluation(trainer, args.num_eval_eps, env_config)
    #
    # Evaluate without Action Masking
    #
    # env_config.update({"mask":False})
    # norm_eval_reward, norm_eval_time, norm_v_total, norm_v_eps = final_evaluation(trainer, args.num_eval_eps, env_config)
    #
    # Data
    #
    avg_train_time = round(np.mean(train_time), 4)
    mask_safe_rolls = 100-(mask_v_eps/args.num_eval_eps*100)
    # norm_safe_rolls = 100-(norm_v_eps/args.num_eval_eps*100)
    # 
    # Print Values
    #
    print("Average Time to Train: ", avg_train_time)
    print("\n-----Evaluation WITH RTA-------")
    print("Average Evaluation Reward with RTA: ", mask_eval_reward)
    print("Number of Safety Violations with RTA: ", mask_v_total)
    print("Percentage of Safe Rollouts with RTA: {}%".format(mask_safe_rolls))
    print("Average Rollout Episode Length with RTA: ", mask_eval_time)
    # print("\n-----Evaluation WITHOUT Action Masking-------")
    # print("Average Evaluation Reward without Masking: ", norm_eval_reward)
    # print("Number of Safety Violations without Masking: ", norm_v_total)
    # print("Percentage of Safe Rollouts without Masking: {}%".format(norm_safe_rolls))
    # print("Average Rollout Episode Length without Masking: ", norm_eval_time)
    #
    # Save Training Data and Agent
    #
    data = {
        "avg_train_time": avg_train_time,
        "train_time": train_time,
        "mask_eval_reward": mask_eval_reward,
        "mask_eval_time": mask_eval_time,
        "mask_safe_rolls": mask_safe_rolls,
        "mask_v_total": mask_v_total,
        # "norm_eval_reward": norm_eval_reward,
        # "norm_eval_time": norm_eval_time,
        # "norm_safe_rolls": norm_safe_rolls,
        # "norm_v_total": norm_v_total,
        "ep_reward": ep_reward,
        "avg_ep_reward": avg_ep_reward,
    }
    with open('results/knapsack_ppo_rta_seeded_results-{}.pkl'.format(str(args.items)), 'wb') as f:
        pickle.dump(data, f)
            


if __name__ == "__main__":
    main()
    
    
