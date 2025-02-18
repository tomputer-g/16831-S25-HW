#!/bin/bash

b=10000
lr=0.0005

echo "Running InvertedPendulum-v4 with batch size $b and learning rate $lr"

python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
    --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b "${b}" -lr "${lr}" -rtg --exp_name "ip_b${b}_lr${lr}_rtg"