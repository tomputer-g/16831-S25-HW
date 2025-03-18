#!/bin/bash
python rob831/hw4_part1/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch1x32 --env_name cheetah-hw4_part1-v0 \
    --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 1 --size 32 \
    --scalar_log_freq -1 --video_log_freq -1 --mpc_action_sampling_strategy 'random'