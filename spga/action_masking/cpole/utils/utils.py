

import numpy as np
from utils.custom_amask_cpole import CartPole
from utils.ga_masking import Agent




def rollout(trainer, pts, env_config={}, ag_type=None, render=False):
    action_masking = env_config.get("use_action_masking", False)
    env = CartPole(env_config)
    eval_rewards = []
    eval_time = []
    v_total = 0
    v_eps = 0
    trajectories = []
    safe_perc = 0.0
    
    for pt in pts:
        obs = env.reset(init=True, state=pt)
        r = 0
        steps = 0
        safe = True
        history = []
        unsafe_actions = 0
        tot_actions = 0
        while True:
            if action_masking:
                pos, vel, theta, theta_vel = obs["actual_obs"]
                history.append(obs["actual_obs"])
            else:
                pos, vel, theta, theta_vel = obs
                history.append(obs)
                
            # Check Safety Criteria
            if pos >= 1.5 or pos <= -1.5:
                if safe:
                    v_eps += 1
                    safe = False
                v_total += 1
                unsafe_actions += 1
            if render:
                env.render()
            if ag_type:
                if ag_type == 'ga':
                    action = trainer.get_action(obs)
                else:
                    action = trainer.compute_single_action(obs)
            else:
                action = trainer.compute_single_action(obs)
            tot_actions += 1
            obs, reward, done, _ = env.step(action)
            r += reward
            steps += 1
            if done:
                safe_perc = (1 - (unsafe_actions/tot_actions))*100
                trajectories.append(history)
                eval_rewards.append(r)
                eval_time.append(steps)
                break
    
    return eval_rewards, eval_time, v_total, v_eps, trajectories, safe_perc