"""

Used to demo a trained agent in the FrozenLake environment.

"""

from ray import tune
import json
import yaml
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import time

from train_ga_frolake_rta import expected_return
from train_ppo_frolake_rta import get_ppo_trainer
from utils.ga_frolake_rta import Agent
from utils.custom_rta_frolake import FrozenLake

def get_args():
    """
    Parse the command arguments
    """
    # create the parser
    parser = argparse.ArgumentParser()
    # Add the arguments to be parsed
    parser.add_argument("--num_rollouts", type=int, default=500, help="Number of times to rollout agent in env")
    parser.add_argument("--render", choices=('True','False'), help="Render the rollout")
    parser.add_argument("--map", type=str, default="8x8", help="The map dimensions of the frozen lake")
    parser.add_argument("--seed", type=int, default=4, help="Indicate the training seed")
    parser.add_argument("--strategy", default="runtime")
    parser.add_argument
    args = parser.parse_args()
    args.render = True if args.render == 'True' else False

    return args

def main():
    args = get_args()
    #
    # Data Organization
    #
    exp = {
        8:{"ga":[], "ppo":[]},
        16:{"ga":[], "ppo":[]},
        32:{"ga":[], "ppo":[]},
    }   
    #
    # Expected Return for all GA Agents
    #
    ga_path = None
    ppo_path = None
    method = "rta"
    # GA
    print("------GA FROLAKE-------")
    for vers in exp.keys():
        args.map = f"{vers}x{vers}"
        env_config = {"use_run_time_assurance": True, "map_name":args.map, "test_mode":True}
        # env = FrozenLake(env_config=env_config)
        agent = Agent()
        agent.load(f"trained_agents/frolake_ga_{method}_seed-{args.seed}_checkpoint-{vers}.json")
        agent.strategy = args.strategy
        eval_rewards, ga_path = expected_return(agent, args.num_rollouts, env_config, agent_type='ga', render=args.render)
        exp[vers]['ga'] = eval_rewards
        print(f"Vers: {vers} || MEAN: {sum(eval_rewards)/args.num_rollouts}")
    #
    # Expected return for all PPO Agents
    #
    print("------PPO FROLAKE-------")
    for vers in exp.keys():
        args.map = f"{vers}x{vers}"
        ppo_agent, _ = get_ppo_trainer(args)
        env_config = {"use_run_time_assurance": True, "map_name":args.map, "test_mode":True}
        name = f"frolake_ppo_{method}_seed-{args.seed}_checkpoint-{vers}"
        ppo_agent.restore("trained_agents/{}/{}".format(name, name))
        eval_rewards, ppo_path = expected_return(ppo_agent, args.num_rollouts, env_config)
        exp[vers]['ppo'] = eval_rewards
        print(f"Vers: {vers} || MEAN: {sum(eval_rewards)/args.num_rollouts}")
    #
    # Save Data
    #
    print('Saving data ...')
    for vers in exp.keys():
        for mode in exp[vers].keys():
            save_name = f"results/expected_return/frolake_{mode}_{method}_seeded_expreturn-{vers}.pkl"
            with open(save_name, 'wb') as f:
                pickle.dump(exp[vers][mode], f)
                
    df = pd.DataFrame(exp)
    df.to_csv('results/expected_return/combined.csv')
        
    return 0
    
    
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("---------------")
    print("Total time: ", abs(end-start))