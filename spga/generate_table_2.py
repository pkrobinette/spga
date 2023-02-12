"""
Reproduce Table 2: Analysis of Self-Preserving Genetic 
Algorithms and Safe Reinforcement Learning

This script utilizes saved training data from each experiment.
"""

import os
import pandas as pd
import numpy as np
import scipy.stats

# Environment Information (Version numbers, names)
ENV_INFO = {
       "cpole": [["1.5", "1.25", "1.0", "0.75", "0.5", "0.25"],"CartPole-v0"],
        "frolake": [["8", "16", "32"], "FrozenLake-v1"],
        "knapsack": [["5", "50", "100"], "Knapsack-v0"],
}

def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate mean confidence interval.

    Parameters
    ----------
    data : list

    Returns
    -------
    m : mean
    c : confidence interval
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def get_data(path):
    """
    Pull data from saved .pkl files from training
    
    Parameters
    ----------
    path : str
        path to each safety mechanism/environment type

    Returns
    -------
    Training Data
    """
    #
    # Get mode, name, and version numbers
    #
    mode = "masking" if "masking" in path else "rta"
    vers = []
    env = None
    if "cpole" in path:
        vers = ["1.5", "1.25", "1.0", "0.75", "0.5", "0.25"]
        env = "cpole"
    elif "frolake" in path:
        vers = ["8", "16", "32"]
        env = "frolake"
    else:
        vers = ["5", "50", "100"] 
        env = "knapsack"
    #
    # Set up data dictionary
    #
    data = {"ga":{}, "ppo":{}}
    #
    # Pull important information
    #
    # for each environment version
    for v in vers:
        # for safety method used in the work
        for method in data.keys():
            #
            # open the saved data as a pandas dataframe
            #
            file = f"{env}_{method}_{mode}_seeded_results-{v}.pkl"
            file_exp = f"expected_return/{env}_{method}_{mode}_seeded_expreturn-{v}.pkl"
            if "ipynb" not in file and "DS_Store" not in file:
                result = {}
                df = pd.read_pickle(path + file)
                df_exp = pd.read_pickle(path+file_exp)
                result["Time to Train"] = round(df["avg_train_time"], 2)
                _, c = mean_confidence_interval(df["train_time"])
                result["Time Cnfd"] = round(c, 0)
                m, c = mean_confidence_interval(df_exp)
                result["Expected Return"] = round(m, 2)
                result["Return Cnfd"] = round(c, 1)
                try:
                    result["Safe Actions"] = round(df["mask_safe_rolls"], 2)
                except:
                    result["Safe Actions"] = round(df["norm_safe_rolls"], 2)
                # save the data to the data dictionary
                data[method][v] = result

    return data


def pretty_print(data):
    """ 
    Print training and testing data as shown in Tab.2 of paper.
    
    Parameters
    ----------
    data : dictionary
    """
    #
    # Print all data
    #
    # for each environment
    print("\n\nAnalysis of Self-Preserving Genetic Algorithms and Safe Reinforcement Learning")
    print("--------------------------------------")
    print(f"Environment | Safety Method |  Variant/Safety Condition |  Training Time (s) |  Expected Return |  Safe Actions (%) |  Training Time (s) |  Expected Return |  Safe Actions (%) | xSpeedUp")
    for env in data.keys():
        # for each method (rta, AM)
        for method in data[env].keys():
            vers, name = ENV_INFO[env]
            info = data[env][method]
            # for each environment version
            for ver in vers:
                speed = round(info["ppo"][ver]["Time to Train"]/info["ga"][ver]["Time to Train"], 2)
                print(f'{name} | {method} | {ver} | {info["ga"][ver]["Time to Train"]}+/-{info["ga"][ver]["Time Cnfd"]} |  {info["ga"][ver]["Expected Return"]}+/-{info["ga"][ver]["Return Cnfd"]} | {info["ga"][ver]["Safe Actions"]}+/-0 | {info["ppo"][ver]["Time to Train"]}+/-{info["ppo"][ver]["Time Cnfd"]} | {info["ppo"][ver]["Expected Return"]} +/-{info["ppo"][ver]["Return Cnfd"]} | {info["ppo"][ver]["Safe Actions"]}+/-0 | {speed}')
        print("-------------------------------------------")
    
    return 0


if __name__ == "__main__":
    #
    # create speedup dictionary to store values
    #
    s_vals = {
        "cpole":{"rta":{}, "am": {}},
        "frolake":{"rta":{}, "am": {}}, 
        "knapsack":{"rta":{}, "am": {}}
    }

    am_path = "./action_masking/"
    rta_path = "./run_time_assurance/"
    
    get_name = lambda x: "rta" if x==rta_path else "am"

    walk = [rta_path, am_path]
    #
    # Calculate and record the geometric mean data
    #
    # environments (cpole, frolake, knapsack)
    for i in s_vals.keys():
        # run time assurance and action masking
        for way in walk:
            path = way+i+"/results/"
            x = get_data(path)
            n = get_name(way)
            s_vals[i][n] = x
        
    pretty_print(s_vals)