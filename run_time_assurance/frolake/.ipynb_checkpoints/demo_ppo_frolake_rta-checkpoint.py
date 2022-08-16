from utils.masking_model import ActionMaskModel, ActionMaskModel_8x8, ActionMaskModel_16x16, ActionMaskModel_32x32
from utils.custom_rta_frolake import FrozenLake

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
from train_ppo_frolake_rta import get_ppo_trainer, final_evaluation


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
    parser.add_argument("--map", type=str, default="8x8", help="The map dimensions of the frozen lake")
    parser.add_argument("--seed", type=int, default=4, help="Indicate the training seed")
    args = parser.parse_args()
    args.render = True if args.render == 'True' else False

    return args


def main():
    args = get_args()
    #
    # Demo With Action Asking
    #
    print("STRATEGY: ", args.strategy)
    mask_agent, mask_env_config = get_ppo_trainer(args)
    mask_env_config["test_mode"] = True
    name = "frolake_ppo_rta_seed-{}_checkpoint-{}".format(args.seed, str(args.map.split("x")[0]))
    mask_agent.restore("trained_agents/{}/{}".format(name, name))
    mask_eval_reward, mask_eval_time, mask_v_total, mask_v_eps, mask_path = final_evaluation(mask_agent, args.num_rollouts, env_config=mask_env_config)
    
    
    print("\n----- Demo With RTA -----")
    print("Avg. Rollout Reward WITH RTA: ", mask_eval_reward)
    print("Avg. Num of Steps WITH RTA: ", mask_eval_time)
    print("Total Violations WITH RTA: ", mask_v_total)
    print("Percentage of Safe Rollouts WITH RTA: {}%".format(100-(mask_v_eps/args.num_rollouts*100)))
    
    #
    # Render Rollout
    #
    print()
    print()
    env = FrozenLake(env_config={"use_run_time_assurance":False, "map_name":args.map})
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