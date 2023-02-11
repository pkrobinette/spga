
from utils.custom_amask_frolake import FrozenLake



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