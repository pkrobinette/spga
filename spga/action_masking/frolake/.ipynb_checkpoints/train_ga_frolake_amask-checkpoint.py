"""
Train genetic algorithm with action masking in FrozenLake.

Notes
-----

"""

from .utils.ga_frolake_masking import Generation
from .utils.custom_amask_frolake import FrozenLake
# from utils.custom_amask_biggerlake import BiggerLakeEnv

import argparse
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


def get_args():
    """
    Parse the command arguments
    """
    # create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments to be parsed
    parser.add_argument("--num_trials", type=int, default=1, help="Number of times to repeat training")
    parser.add_argument("--stop_reward", type=int, default=1, help="Stopping reward criteria for training")
    parser.add_argument("--env_name", type=str, default="frolake", help="Name of the environment")
    parser.add_argument("--strategy", type=str, default='action_masking', help="Training strategy")
    parser.add_argument("--num_eval_eps", type=int, default=20, help="Number of episodes to evaluate the trained agent on after training")
    parser.add_argument("--max_steps", type=int, default=500, help="Max number of generations to train")
    # parser.add_argument("--x_thresh", type=float, default=1.5, help="Action masking threshold used in training")
    parser.add_argument("--map", type=str, default="8x8", help="The grid size of the frozen lake environment")
    # parser.add_argument("--extend", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12, help="Training seed to set randomization for training")
    args = parser.parse_args()

    return args


def final_evaluation(agent, num_rollouts, env_config={}, render=False):
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
                
            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            r += reward
            steps += 1
            if done:
                eval_rewards.append(r)
                eval_time.append(steps)
                break
    
    return np.mean(eval_rewards), np.mean(eval_time), v_total, v_eps, path

def expected_return(agent, num_rollouts, env_config={}, agent_type=None, render=False):
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
            if agent_type == "ga":
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
    
    return eval_rewards, path



def main():
    args = get_args()
    train_time = []
    agent = None
    # Randomly generated seeds for training
    rand_seeds = [4, 36, 27, 2, 98]  # --> for actually training
    for i in range(args.num_trials):
        args.seed = rand_seeds[i]
        env_config = {"use_action_masking": True, "map_name":args.map, "seed":args.seed}
        env = FrozenLake(env_config=env_config)
        agent = Generation(env=env, n=50, g=args.max_steps, solved_score=args.stop_reward, e_rate=0.2, mute_rate=0.3, hidd_l=16, strategy=args.strategy, seed=args.seed)
        print('\n')
        start = time.time()
        agent.simulate()
        end = time.time()
        train_time.append(abs(end-start))
        #
        # Save the agent
        #
        agent.best_agent.save("trained_agents/frolake_ga_masking_seed-{}_checkpoint-{}".format(str(args.seed), str(args.map.split("x")[0])))
    #
    # Evaluate with action masking
    #
    # env_config = {"use_action_masking": True}
    env_config.update({"seed":27})
    mask_eval_reward, mask_eval_time, mask_v_total, mask_v_eps, mask_path = final_evaluation(agent.best_agent, args.num_eval_eps, env_config=env_config)
    #
    # Evaluate without action masking
    #
    env_config.update({"use_action_masking":False})
    agent.best_agent.strategy = None
    norm_eval_reward, norm_eval_time, norm_v_total, norm_v_eps, norm_path = final_evaluation(agent.best_agent, args.num_eval_eps, env_config=env_config)
    #
    # Save Data
    #
    avg_train_time = round(np.mean(train_time), 4)
    mask_safe_rolls = 100-(mask_v_eps/args.num_eval_eps*100)
    norm_safe_rolls = 100-(norm_v_eps/args.num_eval_eps*100)
    data = {
        "avg_train_time": avg_train_time,
        "train_time": train_time,
        "mask_eval_reward": mask_eval_reward,
        "mask_eval_steps": mask_eval_time,
        "mask_safe_rolls": mask_safe_rolls,
        "mask_v_total": mask_v_total,
        "norm_eval_reward": norm_eval_reward,
        "norm_eval_steps": norm_eval_time,
        "norm_safe_rolls": norm_safe_rolls,
        "norm_v_total": norm_v_total,
        "ep_reward": agent.best_fitness,
        "avg_ep_reward": agent.avg_fitness_history,
    }
    
    with open('results/frolake_ga_masking_seeded_results-{}.pkl'.format(str(args.map.split("x")[0])), 'wb') as f:
            pickle.dump(data, f)
    #
    # Print Values
    #
    print("\n FROLAKE GA TRAINING RESULTS")
    print("####################################")
    print("Average Time to Train: ", avg_train_time)
    print("\n-----Evaluation WITH Action Masking-------")
    print("Average Steps to Flag: ", mask_eval_time)
    print("Average Evaluation Reward with Masking: ", mask_eval_reward)
    print("Number of Safety Violations with Masking: ", mask_v_total)
    print("Percentage of Safe Rollouts with Masking: {}%".format(mask_safe_rolls))
    
    print("\n-----Evaluation WITHOUT Action Masking-------")
    print("Average Steps to Flag without Masking: ", norm_eval_time)
    print("Average Evaluation Reward without Masking: ", norm_eval_reward)
    print("Number of Safety Violations without Masking: ", norm_v_total)
    print("Percentage of Safe Rollouts without Masking: {}%".format(norm_safe_rolls))

    #
    # Render Rollout
    #
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
    print(grid)
        
        
    
if __name__ == "__main__":
    main()