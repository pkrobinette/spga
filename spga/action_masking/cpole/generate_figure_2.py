"""
Reproduce Figure 2: CartPole-v0 SPGA-AM vs SRL-AM 50 episode trajectories
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join
import json
from utils.custom_amask_cpole import CartPole
from train_ppo_cpole_amask import get_ppo_trainer
from utils.ga_masking import Agent
from matplotlib.patches import Rectangle
from matplotlib import colors as mcolors
from scipy.signal import savgol_filter


np.random.seed(10)
NUM_TRIALS = 50
NAME = ""
PATH = "action_masking/cpole"
SEED = 4

def rollout(trainer, pts, env_config={}, ag_type=None, render=False):
    action_masking = env_config.get("use_action_masking", False)
    env = CartPole(env_config)
    eval_rewards = []
    eval_time = []
    v_total = 0
    v_eps = 0
    trajectories = []
    
    for pt in pts:
        obs = env.reset(init=True, state=pt)
        r = 0
        steps = 0
        safe = True
        history = []
        while True:
            if action_masking:
                pos, vel, theta, theta_vel = obs["actual_obs"]
                history.append(obs["actual_obs"])
            else:
                pos, vel, theta, theta_vel = obs
                history.append(obs)
                
            # Check Safety Criteria
            if pos >= 1.5 or pos <= -1.5 or theta >= 0.10 or theta <= -0.10:
                if safe:
                    v_eps += 1
                    safe = False
                v_total += 1
            if render:
                env.render()
            if ag_type:
                if ag_type == 'ga':
                    action = trainer.get_action(obs)
                else:
                    action = trainer.compute_single_action(obs)
            else:
                action = trainer.compute_single_action(obs)
            obs, reward, done, _ = env.step(action)
            r += reward
            steps += 1
            if done:
                trajectories.append(history)
                eval_rewards.append(r)
                eval_time.append(steps)
                break
    
    return round(np.mean(eval_rewards),4), round(np.mean(eval_time), 4), v_total, v_eps, trajectories

def generate_plots():    
    exp = {
            1.5:{"ga":{}, "ppo":{}},
            1.25:{"ga":{}, "ppo":{}},
            1.0:{"ga":{}, "ppo":{}},
            0.75:{"ga":{}, "ppo":{}},
            0.5:{"ga":{}, "ppo":{}},
            0.25:{"ga":{}, "ppo":{}},
        }
    
    ga_traj = []
    ppo_traj = []
    env_config = {"use_action_masking": False}
    
    init_pts = [[np.random.uniform(low=-0.05, high=0.05, size=(4,))] for _ in range(NUM_TRIALS)]

    for i, key in enumerate(exp.keys()):
        exp[key]["ga"]["x"] = []
        exp[key]["ga"]["x_dot"] = []
        exp[key]["ga"]["theta"] = []
        exp[key]["ga"]["theta_dot"] = []
        
        print("\nRolling out bound: ", key)
        # rollout ga agent
        ga_agent = Agent()
        ga_agent.load(f"{PATH}/trained_agents/seed_{SEED}/cartpole_ga_masking_seed-{SEED}_checkpoint-{key}.json")
        ga_agent.strategy = None
        for k in range(NUM_TRIALS):
            ga_eval_reward, ga_eval_time, ga_v_total, ga_v_eps, ga_pos = rollout(ga_agent, init_pts[k], env_config, ag_type='ga')
            x, x_dot, theta, theta_dot = zip(*ga_pos[0])
            exp[key]["ga"]["x"].append(x)
            exp[key]["ga"]["x_dot"].append(x_dot)
            exp[key]["ga"]["theta"].append(theta)
            exp[key]["ga"]["theta_dot"].append(theta_dot)
        
        # rollout ppo agent
        exp[key]["ppo"]["x"] = []
        exp[key]["ppo"]["x_dot"] = []
        exp[key]["ppo"]["theta"] = []
        exp[key]["ppo"]["theta_dot"] = []
        ppo_agent, ec = get_ppo_trainer()
        name = f"cartpole_ppo_masking_seed-{SEED}_checkpoint-{key}"
        ppo_agent.restore(f"{PATH}/trained_agents/seed_{SEED}/"+name+"/"+name)
        for k in range(NUM_TRIALS):
            ppo_eval_reward, ppo_eval_time, ppo_v_total, ppo_v_eps, ppo_pos = rollout(ppo_agent, init_pts[k], env_config, ag_type='ppo')
            x, x_dot, theta, theta_dot = zip(*ppo_pos[0])
            exp[key]["ppo"]["x"].append(x)
            exp[key]["ppo"]["x_dot"].append(x_dot)
            exp[key]["ppo"]["theta"].append(theta)
            exp[key]["ppo"]["theta_dot"].append(theta_dot)
    #
    # POSITION PLOT ------------------------------------------------
    #
    print("Creating plots ...")
    sns.set()
    plt.figure(1);
    fig, axis = plt.subplots(2,3, figsize=(15,10));
    axis = axis.flatten();
    
    fig.suptitle('AM Position for CartPole-v0 Rollouts', fontsize=18)
    
    for k, key in enumerate(exp.keys()):
        for i in range(NUM_TRIALS-1):
            axis[k].plot(exp[key]["ga"]['x'][i], np.linspace(0, len(exp[key]["ga"]['x'][i])-1, len(exp[key]["ga"]['x'][i])), 'b')
            axis[k].plot(exp[key]["ppo"]['x'][i], np.linspace(0, len(exp[key]["ppo"]['x'][i])-1, len(exp[key]["ppo"]['x'][i])), 'g--')
        axis[k].plot(exp[key]["ga"]['x'][-1], np.linspace(0, len(exp[key]["ga"]['x'][-1])-1, len(exp[key]["ga"]['x'][-1])),'b', label="ga")
        axis[k].plot(exp[key]["ppo"]['x'][-1], np.linspace(0, len(exp[key]["ppo"]['x'][-1])-1, len(exp[key]["ppo"]['x'][-1])), 'g--', label="ppo")
        axis[k].plot(np.ones((200,))*1.5, np.linspace(0, 199, 200), 'r-.', label='Unsafe')
        axis[k].plot(np.ones((200,))*-1.5, np.linspace(0, 199, 200), 'r-.')
        axis[k].axvspan(-2.4, -1.5, color='red', alpha=0.2);
        axis[k].axvspan(1.5, 2.4, color='red', alpha=0.2);
        axis[k].add_patch(Rectangle((-1.5, 199), 3, 6,
                             facecolor = mcolors.cnames['lime'],
                             alpha=0.5,
                             fill=True, label="Goal"))
        axis[k].set_title("Training Constraint: x={}".format(str(key)))
        axis[k].set_ylabel('Time')
        axis[k].set_xlabel('X Position')
        if k == 0:
            axis[k].text(
                -1.9, 50, "Time", ha="center", va="center", rotation=90, size=15,
                bbox=dict(boxstyle="rarrow, pad=0.25", fc="cyan", ec="b", lw=2))
            axis[k].legend(loc='lower right')
        axis[k].set_xlim([-2.4, 2.4])
        
    plt.tight_layout()
    plt.savefig("artifacts/figure_2a-position.png", bbox_inches='tight', dpi=200)
    #
    # VELOCITY PLOT ------------------------------------------------
    #
#     plt.figure(2);
#     fig, axis = plt.subplots(2,3, figsize=(15,10));
#     axis = axis.flatten();
#     fig.suptitle('AM Velocity for CartPole-v0 Rollouts', fontsize=18)
    
#     for k, key in enumerate(exp.keys()):
#         for i in range(NUM_TRIALS-1):
#             len_ga= len(exp[key]["ga"]['x_dot'][i])
#             len_ppo = len(exp[key]["ppo"]['x_dot'][i])
#             ga_w = ppo_w = 43
#             if  len_ga < ga_w:
#                 ga_w = len_ga-1 if len_ga%2==0 else len_ga
#             if len_ppo < ppo_w:
#                 ppo_w = len_ppo-1 if len_ppo%2==0 else len_ppo
#             axis[k].plot(savgol_filter(exp[key]["ga"]['x_dot'][i], ga_w, 6), np.linspace(0, len_ga-1, len_ga), 'b-')
#             axis[k].plot(savgol_filter(exp[key]["ppo"]['x_dot'][i], ppo_w, 6), np.linspace(0, len_ppo-1, len_ppo), 'g--')
#         # check window size for smoothing
#         len_ga= len(exp[key]["ga"]['x_dot'][-1])
#         len_ppo = len(exp[key]["ppo"]['x_dot'][-1])
#         ga_w = ppo_w = 43
#         if  len_ga < ga_w:
#             ga_w = len_ga-1 if len_ga%2==0 else len_ga
#         if len_ppo < ppo_w:
#             ppo_w = len_ppo-1 if len_ppo%2==0 else len_ppo
#         axis[k].plot(savgol_filter(exp[key]["ga"]['x_dot'][-1], ga_w, 6), np.linspace(0, len_ga-1, len_ga),'b', label="ga")
#         axis[k].plot(savgol_filter(exp[key]["ppo"]['x_dot'][-1], ppo_w, 6), np.linspace(0, len_ppo-1, len_ppo), 'g--', label="ppo")
#         # axis[k].plot(np.ones((200,))*1.5, np.linspace(0, 199, 200), 'r-.', label='Unsafe')
#         # axis[k].plot(np.ones((200,))*-1.5, np.linspace(0, 199, 200), 'r-.')
#         # axis[k].axvspan(-2.4, -1.5, color='red', alpha=0.2);
#         # axis[k].axvspan(1.5, 2.4, color='red', alpha=0.2);
#         axis[k].add_patch(Rectangle((-2.4, 199), 4.8, 6,
#                              facecolor = mcolors.cnames['lime'],
#                              alpha=0.5,
#                              fill=True, label="Goal"))
#         axis[k].set_title("Training Constraint: x={}".format(str(key)))
#         axis[k].set_ylabel('Time')
#         axis[k].set_xlabel('Velocity in X Direction')
#         if k == 0:
#             axis[k].text(
#                 -1.9, 50, "Time", ha="center", va="center", rotation=90, size=15,
#                 bbox=dict(boxstyle="rarrow, pad=0.25", fc="cyan", ec="b", lw=2))
#             axis[k].legend(loc='lower right')
#         axis[k].set_xlim([-2.4, 2.4])
#     plt.tight_layout()
#     plt.savefig("artifacts/cpole_am_x-dot_results{}.png".format(NAME), bbox_inches='tight', dpi=200)
    #
    # THETA PLOT ------------------------------------------------
    #
    plt.figure(3);
    fig, axis = plt.subplots(2,3, figsize=(15,10));
    axis = axis.flatten();
    
    fig.suptitle('AM Theta for CartPole-v0 Rollouts', fontsize=18)
    
    for k, key in enumerate(exp.keys()):
        for i in range(NUM_TRIALS-1):
            axis[k].plot(exp[key]["ga"]['theta'][i], np.linspace(0, len(exp[key]["ga"]['theta'][i])-1, len(exp[key]["ga"]['theta'][i])), 'b')
            axis[k].plot(exp[key]["ppo"]['theta'][i], np.linspace(0, len(exp[key]["ppo"]['theta'][i])-1, len(exp[key]["ppo"]['theta'][i])), 'g--')
        axis[k].plot(exp[key]["ga"]['theta'][-1], np.linspace(0, len(exp[key]["ga"]['theta'][-1])-1, len(exp[key]["ga"]['theta'][-1])),'b', label="ga")
        axis[k].plot(exp[key]["ppo"]['theta'][-1], np.linspace(0, len(exp[key]["ppo"]['theta'][-1])-1, len(exp[key]["ppo"]['theta'][-1])), 'g--', label="ppo")
        axis[k].plot(np.ones((200,))*0.2095, np.linspace(0, 199, 200), 'r-.', label='Unsafe')
        axis[k].plot(np.ones((200,))*-0.2095, np.linspace(0, 199, 200), 'r-.')
        axis[k].axvspan(-0.48, -0.2095, color='red', alpha=0.2);
        axis[k].axvspan(0.2095, 0.48, color='red', alpha=0.2);
        axis[k].add_patch(Rectangle((-0.48, 199), .94, 6,
                             facecolor = mcolors.cnames['lime'],
                             alpha=0.5,
                             fill=True, label="Goal"))
        axis[k].set_title("Training Constraint: x={}".format(str(key)))
        axis[k].set_ylabel('Time')
        axis[k].set_xlabel('Theta Position (rad)')
        if k == 0:
            axis[k].text(
                -.35, 50, "Time", ha="center", va="center", rotation=90, size=15,
                bbox=dict(boxstyle="rarrow, pad=0.25", fc="cyan", ec="b", lw=2))
            axis[k].legend(loc='lower right')
        axis[k].set_xlim([-.48, .48])
        
    plt.tight_layout()
    plt.savefig("artifacts/figure_2b-angle.png", bbox_inches='tight', dpi=200)
    #
    # ANGULAR VELOCITY PLOT ------------------------------------------------
    #
#     plt.figure(2);
#     fig, axis = plt.subplots(2,3, figsize=(15,10));
#     axis = axis.flatten();
#     fig.suptitle('AM Angular Velocity for CartPole-v0 Rollouts', fontsize=18)
    
#     for k, key in enumerate(exp.keys()):
#         for i in range(NUM_TRIALS-1):
#             len_ga= len(exp[key]["ga"]['theta_dot'][i])
#             len_ppo = len(exp[key]["ppo"]['theta_dot'][i])
#             ga_w = ppo_w = 43
#             if  len_ga < ga_w:
#                 ga_w = len_ga-1 if len_ga%2==0 else len_ga
#             if len_ppo < ppo_w:
#                 ppo_w = len_ppo-1 if len_ppo%2==0 else len_ppo
#             axis[k].plot(savgol_filter(exp[key]["ga"]['theta_dot'][i], ga_w, 6), np.linspace(0, len_ga-1, len_ga), 'b-')
#             axis[k].plot(savgol_filter(exp[key]["ppo"]['theta_dot'][i], ppo_w, 6), np.linspace(0, len_ppo-1, len_ppo), 'g--')
#         # check window size for smoothing
#         len_ga= len(exp[key]["ga"]['theta_dot'][-1])
#         len_ppo = len(exp[key]["ppo"]['theta_dot'][-1])
#         ga_w = ppo_w = 43
#         if  len_ga < ga_w:
#             ga_w = len_ga-1 if len_ga%2==0 else len_ga
#         if len_ppo < ppo_w:
#             ppo_w = len_ppo-1 if len_ppo%2==0 else len_ppo
#         axis[k].plot(savgol_filter(exp[key]["ga"]['theta_dot'][-1], ga_w, 6), np.linspace(0, len_ga-1, len_ga),'b', label="ga")
#         axis[k].plot(savgol_filter(exp[key]["ppo"]['theta_dot'][-1], ppo_w, 6), np.linspace(0, len_ppo-1, len_ppo), 'g--', label="ppo")
#         # axis[k].plot(np.ones((200,))*1.5, np.linspace(0, 199, 200), 'r-.', label='Unsafe')
#         # axis[k].plot(np.ones((200,))*-1.5, np.linspace(0, 199, 200), 'r-.')
#         # axis[k].axvspan(-2.4, -1.5, color='red', alpha=0.2);
#         # axis[k].axvspan(1.5, 2.4, color='red', alpha=0.2);
#         axis[k].add_patch(Rectangle((-2.4, 199), 4.8, 6,
#                              facecolor = mcolors.cnames['lime'],
#                              alpha=0.5,
#                              fill=True, label="Goal"))
#         axis[k].set_title("Training Constraint: x={}".format(str(key)))
#         axis[k].set_ylabel('Time')
#         axis[k].set_xlabel('Angular Velocity')
#         if k == 0:
#             axis[k].text(
#                 -1.9, 50, "Time", ha="center", va="center", rotation=90, size=15,
#                 bbox=dict(boxstyle="rarrow, pad=0.25", fc="cyan", ec="b", lw=2))
#             axis[k].legend(loc='lower right')
#         axis[k].set_xlim([-2.4, 2.4])
#     plt.tight_layout()
#     plt.savefig("artifacts/cpole_rta_extended_theta-dot_results-{}.png".format(NAME), bbox_inches='tight', dpi=200)
#     plt.show(); 
    
if __name__ == "__main__":
    generate_plots()