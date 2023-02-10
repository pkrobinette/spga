# Reward shaping needed to train in this environment
python train_ppo_frolake_rta.py --num_trials 5 --map 8x8 --max_steps 500 --num_eval_eps 500 --stop_reward 3.68
python train_ppo_frolake_rta.py --num_trials 5 --map 16x16 --max_steps 500 --num_eval_eps 500 --stop_reward 2
python train_ppo_frolake_rta.py --num_trials 5 --map 32x32 --max_steps 500 --num_eval_eps 500 --stop_reward 5