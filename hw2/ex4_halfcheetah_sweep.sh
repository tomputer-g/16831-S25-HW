#!/bin/bash

# # Array of batch sizes
# batch_sizes=(10000 30000 50000)

# # Array of learning rates
# learning_rates=(0.005 0.01 0.02)

# # Iterate over batch sizes
# for b in "${batch_sizes[@]}"; do
#     # Iterate over learning rates
#     for r in "${learning_rates[@]}"; do
#         echo "Running with batch size $b and learning rate $r"
        
#         python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
#             --discount 0.95 -n 100 -l 2 -s 32 -b "${b}" -lr "${r}" -rtg --nn_baseline \
#             --exp_name "q4_search_b${b}_lr${r}_rtg_nnbaseline" 2>/dev/null

#         echo "----------------------------------------"
#     done
# done

bStar=30000
lrStar=0.02

python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
    --discount 0.95 -n 100 -l 2 -s 32 -b "${bStar}" -lr "${lrStar}" \
    --exp_name "q4_b${b}_r${r}" 2>/dev/null

python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
    --discount 0.95 -n 100 -l 2 -s 32 -b "${bStar}" -lr "${lrStar}" -rtg \
    --exp_name "q4_b${b}_r${r}_rtg" 2>/dev/null

python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
    --discount 0.95 -n 100 -l 2 -s 32 -b "${bStar}" -lr "${lrStar}" --nn_baseline \
    --exp_name "q4_b${b}_r${r}_nnbaseline" 2>/dev/null

python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
    --discount 0.95 -n 100 -l 2 -s 32 -b "${bStar}" -lr "${lrStar}" -rtg --nn_baseline \
    --exp_name "q4_b${b}_r${r}_rtg_nnbaseline" 2>/dev/null


# echo "Done."
