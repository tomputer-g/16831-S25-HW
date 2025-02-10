#!/bin/bash

run_agent() {
    local agent=$1

    echo "Running agent: $agent with 2 expert trajs"
    # String manipulation
    local agent_lower="${agent,,}"          # Convert to lowercase (ant)
    local exp_name="bc_${agent_lower}_q1_2"    # Add prefix (bc_ant)
    local exp_policy_file="rob831/policies/experts/${agent}.pkl"
    local exp_data="rob831/expert_data/expert_data_${agent}-v2.pkl"
    
    # Example config path using lowercase
    local config_path="configs/${agent_lower}.yaml"
    
    # Run command with agent-specific configuration
    output=$(python rob831/scripts/run_hw1.py \
        --expert_policy_file "$exp_policy_file" \
        --env_name "$agent-v2" --exp_name "$exp_name" --n_iter 1 \
        --expert_data "$exp_data" \
        --train_batch_size 100 \
        --num_agent_train_steps_per_iter 1000 \
        --video_log_freq -1 \
        --ep_len 1000 --eval_batch_size 2000 \
        --n_layers 2 2>/dev/null)    

    # Extract values
    train_avg=$(echo "$output" | grep "Train_AverageReturn" | awk '{print $3}')
    train_std=$(echo "$output" | grep "Train_StdReturn" | awk '{print $3}')
    
    # Print Train avg/std
    echo "Train: ${train_avg}/${train_std}"
    echo "---------------------------------"
}

# Example usage with different agents
for agent in Ant HalfCheetah Hopper Humanoid Walker2d; do
    run_agent "$agent"
done