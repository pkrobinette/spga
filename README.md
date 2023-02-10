# Self-Preserving Genetic Algorithms for Safe Learning in Discrete Action Spaces
This code base implements self-preserving genetic algorithms (SPGA) and safe reinforcement learning (SRL) and provides a comparison of the safe learning methods in three different discrete action environments:
1. CartPole-v0 (OpenAI Gym)
2. FrozenLake-v1 (OpenAI Gym)
3. Knapsack-v0 (OR-Gym)

## Structure
The code base is structured as follows:
```bash
.
├── action_masking
│   ├── cpole           # CartPole-v0
│   ├── frolake         # FrozenLake-v1
│   └── knapsack        # Knapsack-v0
└── run_time_assurance
    ├── cpole           # CartPole-v0
    ├── frolake         # FrozenLake-v1
    └── knapsack        # Knapsack-v0
 ```
    
## Installation
1. Navigate to the AAAI_SPGA directory
2. Run:
```bash
conda env create -f environment.yml
conda activate spga
```
 
## Replication
### xSpeedUp
To reproduce the xSpeedUp results shown in Table 2, run the `Generate Table 2.ipynb` file (~30 sec.).

### Agent Rollout Plots
To reproduce the rollout plots shown in the graph, navigate to the respective directory and run the respective` *plot*.ipynb` file. For instance, if you want to reproduce the CartPole-v0 action masking plots, navigate to `action_masking` > `cpole` directory and run `plot_cpole_amask_rollouts.ipynb`. The time to plot is environment dependent. Each plot .ipynb, however, takes approximately less than 10 min. to run.

### Agent Training (not necessary)
To reproduce the training of each SPGA and SRL agent in their respective environment, navigate to the environment and method in question and run the `./train_spga.sh` or `./train_srl.sh` bash script. This will run the safe learning method for each version of that environment. Time to train is environment dependent. Estimates of total training time are shown below. Because the training can take a long time, the trained agents for each respective test seed (SEED 4) are provided in each directory. Seed implementations used in this work during training are automatically integrated into the code.

*Note:* These are very rough estimates
| Environment | Safe Learning Method | Time to Run File |
|-------------|------------------|----------------------|
|CartPole-v0  | SPGA                 |    < 10 min       |
|CartPole-v0  | SRL                  |    < 20 min      |
|FrozenLake-v1  | SPGA                 |    < 1 hr      |
|FrozenLake-v1  | SRL                  |    <  4 hr      |
|Knapsack-v0  | SPGA                 |    < 2 hr       |
|FrozenLake-v1  | SRL                  |    <  7 hr      |

### Computational Resources
These experiments were conducted on an 2.3 GHz 8-Core Intel Core i9 processor with 16 GB 2667 MHz DDR4 of memory.

## Notes
- When running a bash script, if you get a similar error, make sure to change the executable permissions of the file:
    - ```permission denied: ./test.sh```
    - Run: 
    ```bash
    chmod +x test.sh
    ```
