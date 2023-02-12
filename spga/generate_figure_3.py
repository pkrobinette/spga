"""
Reproduce Figure 3:  FrozenLake-v1 SPGA-Action Mask (SPGA-AM) vs. SRL-Action Mask (SRL-AM)
"""


from action_masking.frolake.utils.custom_amask_frolake import FrozenLake
from action_masking.frolake.train_ga_frolake_amask import expected_return
from action_masking.frolake.train_ppo_frolake_amask import get_ppo_trainer
from action_masking.frolake.utils.ga_frolake_masking import Agent

from ray import tune
import json
import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import time


def get_args():
    """
    Parse the command arguments
    """
    # create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments to be parsed
    parser.add_argument("--render", action='store_true')
    parser.add_argument('--no-render', dest='render', action='store_false')
    parser.set_defaults(render=True)
    parser.add_argument("--map", type=str, default="16x16", help="The map dimensions of the frozen lake")
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()

    return args


def rollout(agent, num_rollouts, env_config={}, ag_type=None, render=False):
    """
    Used for final evaluation policy rollout
    """
    action_masking = env_config.get("use_action_masking", False)
    env = FrozenLake(env_config)
    eval_rewards = []
    eval_time = []
    v_total = 0
    v_eps = 0
    
    for _ in range(num_rollouts):
        safe = True
        steps = 0
        r = 0
        obs = env.reset()
        path = []
        while True:
            if render:
                env.render()
            if action_masking:
                s = obs["state"]
            else:
                s = obs
            path.append(s)
            if s in env.holes:
                if safe:
                    v_eps += 1
                    safe = False
                v_total += 1
            if ag_type == 'ga':
                action = agent.get_action(obs)
            else:
                action = agent.compute_single_action(obs)
            obs, reward, done, _ = env.step(action)
            r += reward
            steps += 1
            if done:
                eval_rewards.append(r)
                eval_time.append(steps)
                break
    
    return eval_rewards, eval_time, v_total, v_eps, path


def show_path(path, vers, env_config, name):
    """
    
    Parameters
    ----------
    path : list
        path taken by the agent
    vers : int
        size of the map
    env_config : dict
        environment configurations
    name : str
        name to be printed on top of path
    """
    env = FrozenLake(env_config=env_config)
    print()
    if name:
        print(f"{name} {vers}x{vers} Path")
    print("------------------------------\n")
    grid_s = int(vers)
    grid = [0]*grid_s**2
    grid[-1] = "G"
    for h in env.holes:
        grid[h] = "H"
        
    for p in path:
        grid[p] = "#"
    
    grid = np.array(grid)
    grid = grid.reshape(grid_s, grid_s)
    
    file = open('artifacts/frolake_'+name+'-AM_path.txt','w')
    for i in range(len(grid)):
        line = "".join(grid[i])
        print(line)
        file.write(line)
        file.write("\n")
    file.close()
    
    return 0

def main():
    args = get_args()
    assert (args.map in ["32x32", "16x16", "8x8"]), "incorrect map name; try again."
    vers = args.map.split("x")[0]
    #
    # Generate path for GA agent
    #
    env_config = {"use_action_masking": True, "map_name":args.map}
    agent = Agent()
    agent.load(f"action_masking/frolake/trained_agents/frolake_ga_masking_seed-4_checkpoint-{vers}.json")
    agent.strategy = "action_masking"
    eval_rewards, ga_path = expected_return(agent, 1, env_config, agent_type='ga', render=args.render)
    #
    # Generate path for PPO Agent
    #
    ppo_agent, ec = get_ppo_trainer(args)
    name = f"frolake_ppo_masking_seed-4_checkpoint-{vers}"
    ppo_agent.restore("action_masking/frolake/trained_agents/{}/{}".format(name, name))
    eval_rewards, ppo_path = expected_return(ppo_agent, 1, ec)
    #
    # Show Paths
    #
    show_path(ga_path, vers, env_config, "SPGA")
    show_path(ppo_path, vers, env_config, "SRL")
    
    return 0
    
    
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("---------------")
    print("Total time: ", abs(end-start))