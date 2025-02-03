#!/bin/bash
#Thanks, Perplexity#!/bin/bash

run_agent() {
    local agent=$1
    echo "Running agent: $agent"
    # String manipulation examples
    local agent_lower="${agent,,}"          # Convert to lowercase (ant)
    local exp_name="bc_${agent_lower}"    # Add prefix (bc_ant)
    local exp_policy_file="rob831/policies/experts/${agent}.pkl"
    local exp_data="rob831/expert_data/expert_data_${agent}-v2.pkl"
    
    # Example config path using lowercase
    local config_path="configs/${agent_lower}.yaml"
    
    # Run command with agent-specific configuration
    output=$(python rob831/scripts/run_hw1.py \
        --expert_policy_file "$exp_policy_file" \
        --env_name "$agent-v2" --exp_name "$exp_name" --n_iter 1 \
        --expert_data "$exp_data" \
        --train_batch_size 500 \
        --num_agent_train_steps_per_iter 2000 \
        --video_log_freq -1 \
        --n_layers 2 2>/dev/null)    
    # Extract values
    eval_avg=$(echo "$output" | grep "Eval_AverageReturn" | awk '{print $3}')
    initial_avg=$(echo "$output" | grep "Initial_DataCollection_AverageReturn" | awk '{print $3}')

    # Calculate percentage
    percentage=$(awk "BEGIN {printf \"%.3f\", ($eval_avg / $initial_avg) * 100}")
    
    # Print agent name with result
    echo "${agent}: ${percentage}%"
}

# Example usage with different agents
for agent in Ant HalfCheetah Hopper Humanoid Walker2d; do
    run_agent "$agent"
done