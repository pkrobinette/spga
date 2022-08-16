from utils.masking_model import ActionMaskModel
from utils.custom_rta_cpole import CartPole

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
from train_ppo_cpole_rta import get_ppo_trainer


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

def rollout(trainer, pts, env_config={}, render=False):
    """
    Rollout the trainer
    """
    action_masking = True if env_config["use_action_masking"] == True else False
    env = CartPole(env_config)
    eval_rewards = []
    eval_time = []
    v_total = 0
    v_eps = 0
    trajectories = []
    actions = []
    
    for pt in pts:
        obs = env.reset(init=True, state=pt)
        r = 0
        steps = 0
        safe = True
        x_values = []
        while True:
            if action_masking:
                pos, vel, theta, theta_vel = obs["actual_obs"]
            else:
                pos, vel, theta, theta_vel = obs
            x_values.append(pos)
            # Check Safety Criteria
            if pos > 1.5 or pos < -1.5:
                if safe:
                    v_eps += 1
                    safe = False
                v_total += 1
            if render:
                env.render()
            action = trainer.compute_single_action(obs)
            actions.append(action)
            obs, reward, done, _ = env.step(action)
            r += reward
            steps += 1
            if done:
                trajectories.append(x_values)
                eval_rewards.append(r)
                eval_time.append(steps)
                break
    
    return round(np.mean(eval_rewards),4), round(np.mean(eval_time), 4), v_total, v_eps, trajectories

def main():
    args = get_args()
    #
    # Demo
    #
    init_pts = [np.random.uniform(low=-0.05, high=0.05, size=(4,)) for _ in range(args.num_rollouts)]
    agent, env_config = get_ppo_trainer(args)
    name = "cartpole_ppo_rta_seed-{}_checkpoint-{}".format(args.seed, args.x_thresh)
    agent.restore("trained_agents/seed_{}/{}/{}".format(args.seed, name, name))
    eval_reward, eval_time, v_total, v_eps, traj = rollout(mask_agent, pts=init_pts, env_config=mask_env_config, render=args.render)
    
    print("\n------ Demo ------")
    print("Avg. Rollout Reward: ", eval_reward)
    print("Avg. Rollout Time: ", eval_time)
    print("Total Violations: ", v_total)
    print("Percentage of Safe Rollouts: {}%".format(100-(v_eps/args.num_rollouts*100)))
    
    
if __name__ == "__main__":
    main()