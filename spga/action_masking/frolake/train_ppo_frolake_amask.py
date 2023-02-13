"""
Train agent with PPO using action masking in the CartPole-v0 environment.

Tuned Hyperparameters From : https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/cartpole-ppo.yaml
"""

from utils.masking_model import ActionMaskModel, ActionMaskModel_8x8, ActionMaskModel_16x16, ActionMaskModel_32x32
from utils.custom_amask_frolake import FrozenLake

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

import sys
sys.path.append(sys.path[0]+"/results")
sys.path.append(sys.path[0]+"/trained_agents")

def get_args():
    """
    Parse the command arguments
    """
    # create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments to be parsed
    parser.add_argument("--num_trials", type=int, default=1, help="Number of times to repeat training")
    parser.add_argument("--stop_reward", type=int, default=0.95, help="Stopping reward criteria for training")
    parser.add_argument("--env_name", type=str, default="frolake", help="Name of the environment")
    parser.add_argument("--strategy", type=str, default='action_masking', help="Training strategy")
    parser.add_argument("--num_eval_eps", type=int, default=20, help="Number of episodes to evaluate the trained agent on after training")
    parser.add_argument("--max_steps", type=int, default=500, help="Max number of training steps during the training process")
    parser.add_argument("--map", type=str, default="8x8", help="The grid size of the frozen lake environment")
    parser.add_argument("--seed", type=int, default=12, help="Training seed to set randomization for training")
    args = parser.parse_args()

    return args


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
    config = ppo.DEFAULT_CONFIG.copy()
    # tuned hyperparameters for
    config["env"] = FrozenLake
    config["env_config"] = {"use_action_masking": False}
    
    if args:
        if vars(args).get("strategy", "action_masking"):
            print("Using action masking to train ...")
            if args.map == "32x32":
                ModelCatalog.register_custom_model("kp_mask", ActionMaskModel_32x32)
            elif args.map == "8x8":
                ModelCatalog.register_custom_model("kp_mask", ActionMaskModel_8x8)
            elif args.map == "16x16":
                ModelCatalog.register_custom_model("kp_mask", ActionMaskModel_16x16)
            else:
                ModelCatalog.register_custom_model("kp_mask", ActionMaskModel)
            config["env_config"] = {"use_action_masking": True, "map_name":args.map}
            config["model"] = {
                "custom_model": "kp_mask",
            }
            
    if vars(args).get("map"):
        print("Using Map: ", args.map)
        config["env_config"]["map_name"] = args.map
        
    config["env_config"]["seed"] = args.seed
    # config.update({'seed': args.seed})
    
    trainer = ppo.PPOTrainer(config=config)
    return trainer, config["env_config"]


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
    """
    action_masking = env_config.get("use_action_masking", False)
    env = FrozenLake(env_config)
    eval_rewards = []
    eval_time = []
    v_total = 0
    v_eps = 0
    
    for _ in range(n_final_eval):
        obs = env.reset()
        r = 0
        steps = 0
        safe = True
        path = []
        while True:
            if action_masking:
                s = obs["state"]
            else:
                s = obs
            path.append(s)
            # Check Safety Criteria
            if s in env.holes:
                if safe:
                    v_eps += 1
                    safe = False
                v_total += 1
            
            action = trainer.compute_single_action(obs)
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
    #
    # Setup and seeds
    #
    args = get_args()
    ray.shutdown()
    ray.init()
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
        name = "frolake_ppo_masking_seed-{}_checkpoint-{}".format(str(args.seed), str(args.map.split("x")[0]))
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
    print('action masking')
    mask_eval_reward, mask_eval_time, mask_v_total, mask_v_eps, mask_path = final_evaluation(trainer, args.num_eval_eps, env_config)
    #
    # Evaluate without Action Masking
    #
    print('not action masking')
    args.strategy = None
    agent, env_config = get_ppo_trainer(args)
    print(env_config)
    agent.restore(checkpoint)
    norm_eval_reward, norm_eval_time, norm_v_total, norm_v_eps, norm_path = final_evaluation(agent, args.num_eval_eps, env_config)
    #
    # Data
    #
    avg_train_time = round(np.mean(train_time), 4)
    mask_safe_rolls = 100-(mask_v_eps/args.num_eval_eps*100)
    norm_safe_rolls = 100-(norm_v_eps/args.num_eval_eps*100)
    # 
    # Print Values
    #
    print("Average Time to Train: ", avg_train_time)
    print("\n-----Evaluation -------")
    print("Average Evaluation Reward: ", mask_eval_reward)
    print("Number of Safety Violations: ", mask_v_total)
    print("Percentage of Safe Rollouts: {}%".format(mask_safe_rolls))
    print("Average Steps: ", mask_eval_time)
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
        "norm_eval_reward": norm_eval_reward,
        "norm_eval_time": norm_eval_time,
        "norm_safe_rolls": norm_safe_rolls,
        "norm_v_total": norm_v_total,
        "ep_reward": ep_reward,
        "avg_ep_reward": avg_ep_reward,
    }
    with open('results/frolake_ppo_masking_seeded_results-{}.pkl'.format(str(args.map.split("x")[0])), 'wb') as f:
        pickle.dump(data, f)
            
            
    #
    # Render Rollout
    #
    print()
    print()
    env = FrozenLake(env_config={"use_action_masking":True, "map_name":args.map})
    g_size = int(args.map.split("x")[0])
    grid = [0]*g_size**2
    grid[-1] = "G"
    for h in env.holes:
        grid[h] = "H"
        
    for p in mask_path:
        grid[p] = "#"
    
    grid = np.array(grid)
    grid = grid.reshape(g_size, g_size)
    print(grid)
            


if __name__ == "__main__":
    main()
    
    
