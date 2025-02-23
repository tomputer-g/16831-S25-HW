#!/bin/bash

lambdas=(0 0.95 0.99 1)

for l in "${lambdas[@]}"; do
    python rob831/scripts/run_hw2.py --env_name Hopper-v4 --ep_len 1000 --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 \
        --reward_to_go --nn_baseline --action_noise_std 0.5 --gae_lambda "${l}" \
        --exp_name "q5_b2000_r0.001_lambda${l}" 2>/dev/null
    echo "----------------------------------------"
done

# python rob831/scripts/run_hw2.py --env_name Hopper-v4 --ep_len 1000 --discount 0.99 -n 20 -l 2 -s 32 -b 200 -lr 0.001 \
#         --reward_to_go --nn_baseline --action_noise_std 0.5 --gae_lambda 1 \
#         --exp_name q5_b2000_r0.001_lambda1
# echo "----------------------------------------"

# python rob831/scripts/run_hw2.py --env_name Hopper-v4 --ep_len 1000 --discount 0.99 -n 20 -l 2 -s 32 -b 200 -lr 0.001 \
#         --reward_to_go --nn_baseline --action_noise_std 0.5 \
#         --exp_name q5_b2000_r0.001_vanillabaseline