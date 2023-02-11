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

from train_ga_frolake_amask import final_evaluation, expected_return
from utils.ga_frolake_masking import Agent
from utils.custom_amask_frolake import FrozenLake

def get_args():
    """
    Parse the command arguments
    """
    # create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments to be parsed
    parser.add_argument("--num_rollouts", type=int, default=1, help="Number of times to rollout agent in env")
    parser.add_argument("--render", choices=('True','False'), help="Render the rollout")
    parser.add_argument("--map", type=str, default="8x8", help="The map dimensions of the frozen lake")
    parser.add_argument("--seed", type=int, default=4, help="Indicate the training seed")
    args = parser.parse_args()
    args.render = True if args.render == 'True' else False

    return args

def main():
    args = get_args()
    # assert seed is a valid entry
    assert (args.seed in [98, 4, 36, 27, 2]),"Not a valid training seed"
    
    #
    # Demo With Action Asking
    #
    env_config = {"use_action_masking": True, "map_name":args.map}
    env = FrozenLake(env_config=env_config)
    agent = Agent()
    agent.load("trained_agents/frolake_ga_masking_seed-{}_checkpoint-{}.json".format(args.seed, str(args.map.split("x")[0])))
    agent.strategy = "action_masking"
    mask_eval_reward, mask_eval_time, mask_v_total, mask_v_eps, mask_path = final_evaluation(agent, args.num_rollouts, env_config, render=args.render)
    #
    # Demo Without Action Masking
    #
    env_config.update({"use_action_masking": False})
    agent.strategy = None
    norm_eval_reward, norm_eval_time, norm_v_total, norm_v_eps, norm_path = final_evaluation(agent, args.num_rollouts, env_config)
    
    print("\n----- Demo With Masking -----")
    print("Avg. num of steps to goal: ", mask_eval_time)
    print("Avg. Rollout Reward WITH Masking: ", mask_eval_reward)
    print("Total Violations WITH Masking: ", mask_v_total)
    print("Percentage of Safe Rollouts WITH Masking: {}%".format(100-(mask_v_eps/args.num_rollouts*100)))
    print("\n------ Demo WITHOUT Masking ------")
    print("Avg. num steps to goal: ", norm_eval_time)
    print("Avg. Rollout Reward: ", norm_eval_reward)
    print("Total Violations: ", norm_v_total)
    print("Percentage of Safe Rollouts: {}%".format(100-(norm_v_eps/args.num_rollouts*100)))
    
    print()
    print()
    g_size = int(args.map.split("x")[0])
    grid = [0]*g_size**2
    grid[-1] = "G"
    for h in env.holes:
        grid[h] = "H"
        
    for p in mask_path:
        grid[p] = "#"
    
    grid = np.array(grid)
    grid = grid.reshape(g_size, g_size)
    for i in range(len(grid)):
        print(grid[i])
    
    
    
    
if __name__ == "__main__":
    main()