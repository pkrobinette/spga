"""
Reproduce Table 3: Geometric Mean xSpeedUp of SPGA vs SRL

This script utilizes saved training data to calculate the geometric
mean xSpeedUp data of SPGA vs SRL.
"""

import os
import pandas as pd
import numpy as np


def g_mean(x):
    """
    Calculate the geometric mean
    
    Parameters
    ----------
    x : list
        A list of ratios for each training environment version.

    Returns:
    Geometric mean of the list. 
    """
    a = np.log(x)
    return np.exp(a.mean())

def get_speedup(path):
    """
    Calculate the xSpeedUp of using spga vs safe-rl.
    
    Parameters
    ----------
    path : str
        path to each safety mechanism/environment type

    Returns
    -------
    Geometric mean of the recorded time to train data.
    """
    #
    # Get the file paths
    #
    files = [f for f in os.listdir(path) if ".pkl" in f]
    #
    # Set up data dictionary
    #
    data = {"ga":{}, "ppo":{}}
    
    for file in files:
        #
        # open the saved data as a pandas dataframe
        #
        if "ipynb" not in file and "DS_Store" not in file:
            # ppo or ga
            name = file.split("_")[1]
            if "cpole" in path:
                size = file.split("-")[-1].replace(".pkl", "")
            else:
                size = file.split("-")[-1].split(".")[0]
            result = {}
            df = pd.read_pickle(path + file)
            result["Time to Train"] = round(df["avg_train_time"], 2)
            result["Time Std"] = round(np.std(df["train_time"]), 0)
            # result["Expected Return"] = round(df["mask_eval_reward"], 2)
            # result["Return Std"] = round(np.std(df["ep_reward"]), 0)
            # result["Safe Actions"] = round(df["mask_safe_rolls"], 2)
            data[name][size] = result

    #
    # Collect all ratios of ppo to ga training times
    # 
    s_stat = []
    if "cpole" in path:
        for i in ["1.5", "1.25", "1.0", "0.75", "0.5", "0.25"]:
            s_stat.append(round(data["ppo"][i]["Time to Train"]/data["ga"][i]["Time to Train"], 2))
    elif "frolake" in path:
        for i in ["8", "16", "32"]:
            s_stat.append(round(data["ppo"][i]["Time to Train"]/data["ga"][i]["Time to Train"], 2))
    else:
        for i in ["5", "50", "100"]:
            s_stat.append(round(data["ppo"][i]["Time to Train"]/data["ga"][i]["Time to Train"], 2))
    
    return round(g_mean(s_stat), 2) 


if __name__ == "__main__":
    #
    # create speedup dictionary to store values
    #
    s_vals = {
        "cpole":{},
        "frolake":{}, 
        "knapsack":{}
    }
    #
    # gather file paths
    #
    am_path = "./action_masking/"
    rta_path = "./run_time_assurance/"
    
    walk = [rta_path, am_path]
    all_envs = []
    #
    # Calculate and record the geometric mean data
    #
    print("Geometric Mean xSpeedUp of SPGA vs. SRL")
    print("--------------------------------------")
    for i in s_vals.keys():
        for way in walk:
            path = way+i+"/results/"
            x = get_speedup(path)
            all_envs.append(x)
            print("{} - {} SpeedUp: {}".format(i, "AM" if way == am_path else "RTA", x))
    #
    # Print the information
    #
    print("All Envs: {}".format(round(g_mean(all_envs), 2)))