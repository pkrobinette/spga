#!/bin/bash
# generate table 2: analysis
printf "\nGenerating Table 2...\n\n"
python generate_table_2.py

# generate table 3: geometric mean values
printf "\n\nGenerating Table 3...\n\n"
python generate_table_3.py

# generate figure 2: cartpole am
printf "\n\nGenerating Figure 2...\n\n"
python action_masking/cpole/generate_figure_2.py

# generate figure 3: frozenlake am
printf "\n\nGenerating Figure 3...\n\n"
python action_masking/frolake/generate_figure_3.py

# generate figure 4: knapsack rta
printf "\n\nGenerating Figure 4...\n\n"
python run_time_assurance/knapsack/generate_figure_4.py


printf "\n\n For all produced artifacts, see the 'artifacts' directory\n\n"