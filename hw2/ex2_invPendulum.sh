#!/bin/bash

# b=10000
# lr=0.0005

# echo "Running InvertedPendulum-v4 with batch size $b and learning rate $lr"

# python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
#     --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b "${b}" -lr "${lr}" -rtg --exp_name "ip_b${b}_lr${lr}_rtg"

# # Array of batch sizes
batch_sizes=(5000 10000 20000)

# # Array of learning rates
learning_rates=(0.005 0.01 0.02)

# # Iterate over batch sizes
for b in "${batch_sizes[@]}"; do
    # Iterate over learning rates
    for r in "${learning_rates[@]}"; do
        echo "Running with batch size $b and learning rate $r"
        
        python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
            --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b "${b}" -lr "${r}" -rtg --exp_name "q2_b${b}_r${r}"


        echo "----------------------------------------"
    done
done