# Self-Preserving Genetic Algorithms for Safe Learning in Discrete Action Spaces
This code base implements self-preserving genetic algorithms (SPGA) and safe reinforcement learning (SRL) and provides a comparison of the safe learning methods in three different discrete action environments:
1. CartPole-v0 (OpenAI Gym)
2. FrozenLake-v1 (OpenAI Gym)
3. Knapsack-v0 (OR-Gym)

## Structure
The code base is structured as follows:
```bash
.
├── Dockerfile              # Dockerfile for environment
├── README.md
├── environment.yml         # conda environment file
├── setup.py
└── spga                    # artifiact reproduction directory
    ├── artifacts           # Saving location for reproduced artifacts
    │   └── logs
    ├── generate_artifacts.sh       # SCRIPT TO REPRODUCE ALL ARTIFACTS
    ├── generate_table_2.py
    ├── generate_table_3.py
    ├── action_masking      
    │   ├── cpole           # CartPole-v0 Action Masking
    │   ├── frolake         # FrozenLake-v1 Action Masking
    │   └── knapsack        # Knapsack-v0 Action Masking
    └── run_time_assurance
        ├── cpole           # CartPole-v0 Run Time Assurance
        ├── frolake         # FrozenLake-v1 Run Time Assurance
        └── knapsack        # Knapsack-v0 Run Time Assurance
 ```
    
## Installation
Clone this repository to your local machine.
```bash
git clone https://github.com/pkrobinette/spga.git
git checkout 5d12d14
cd spga
```

### Option 1: Docker (recommended for ICCPS'23 AE) (~5 min.)
1. Build the Docker image
```bash
sudo docker build . -t spga_image
```
2. Create and run a container of the Docker image
```bash
sudo docker run --name spga --rm -it spga_image bash
```

### Option 2: Local Install
1. Create a conda environment
```bash
conda env create -f environment.yml
```
2. Activate the conda environment
```bash
conda activate spga
```

## ICCPS '23 Artifact Evaluation
This artifact is intended to reproduce each of the plots, figures, training, and testing results included in the corresponding ICCPS '23 paper. 

### Tables and Figures
The tables and figures reproduced in this artifiact evaluation are listed below.

1. `Table 2: Analysis of Self-Preserving Genetic Algorithms and Safe Reinforcement Learning`

2. `Table 3: Geometric Mean xSpeedUp of SPGA vs. SRL` 

3. `Figure 2: CartPole-v0 SPGA-AM vs. SRL-AM Results`

4. `Figure 3: FrozenLake-v1 SPGA-AM vs. SRL-AM Results`

5. `Figure 4: Knapsack-v0 SPGA-RTA vs. SRL-RTA`

### Instructions to Reproduce (~4 min.)
1. Navigate to the `spga.spga` directory. 
```bash
cd spga
```
2. Make the `generate_artifacts.sh` script executable
```bash
chmod +x generate_artifacts.sh
```
3. Run the script
```bash
./generate_artifacts.sh
```
4. *(If Using Docker)* Pull artifacts the docker image.

    a. Open a seperate window on your local machine and navigate to the embedded sgpa folder. (spga.spga)
    
    b. Make the `retrieve.sh` script executable
    ```bash
    chmod +x retrieve.sh
    ```
    
    c. Run the script
    ```bash
    ./retrieve.sh
    ```
    
5. Evaluate reproduced artifacts.

### Agent Training (~72 hrs.)
To reproduce the training of each SPGA and SRL agent in their respective environment, navigate to the environment and method in question and run the `./train.sh` bash script. This will run the safe learning method for each version of that environment. Time to train is environment dependent. It takes approximately 72 hours to train all agents, for each environment, and each seed. Because the training can take a long time, the trained agents for each respective test seed (SEED 4) are provided in each directory. Seed implementations used in this work during training are automatically integrated into the code.

### Time Estimates
*Note:* These are very rough estimates
| Command | Time to Run File |
|-------------|------------------|
|Dockerfile Setup  | ~5 min.  |
|Reproduce Artifacts  | ~4 min. |
|Train All Agents  | ~72 hrs. |


### Computational Resources
These experiments were conducted on an 2.3 GHz 8-Core Intel Core i9 processor with 16 GB 2667 MHz DDR4 of memory.

## Notes
- When running a bash script, if you get a similar error, make sure to change the executable permissions of the file:
    - ```permission denied: ./test.sh```
    - Run: 
    ```bash
    chmod +x test.sh
    ```
