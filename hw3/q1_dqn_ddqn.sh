#!/bin/bash

# DQN
seeds=(1 2 3)
for s in "${seeds[@]}"; do
    python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name "q1_dqn_${s}" --seed "${s}" --no_gpu
done

# DDQN

for s in "${seeds[@]}"; do
    python rob831/scripts/run_hw3_dqn.py --env_name LunarLander-v3 --exp_name "q1_doubledqn_${s}" --double_q --seed "${s}" --no_gpu
done