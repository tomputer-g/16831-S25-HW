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

    #Example of output:
    # Collecting data for eval...
    # /home/tomg/src/16831-S25-HW/hw1/env/lib/python3.10/site-packages/gym/utils/passive_env_checker.py:241: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
    #   if not isinstance(terminated, (bool, np.bool8)):
    # Eval_AverageReturn : 341.92864990234375
    # /home/tomg/src/16831-S25-HW/hw1/env/lib/python3.10/site-packages/tensorboardX/summary.py:153: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
    #   scalar = float(scalar)
    # Eval_StdReturn : 102.69835662841797
    # Eval_MaxReturn : 569.781982421875
    # Eval_MinReturn : 223.09657287597656
    # Eval_AverageEpLen : 63.125
    # Train_AverageReturn : 10344.517578125
    # Train_StdReturn : 20.9814453125
    # Train_MaxReturn : 10365.4990234375
    # Train_MinReturn : 10323.5361328125
    # Train_AverageEpLen : 1000.0
    # Train_EnvstepsSoFar : 0
    # TimeSinceStart : 2.883723020553589
    # Training Loss : 0.013900777325034142
    # Initial_DataCollection_AverageReturn : 10344.517578125
    # Done logging...
    # Extract values
    train_avg=$(echo "$output" | grep "Train_AverageReturn" | awk '{print $3}')
    train_std=$(echo "$output" | grep "Train_StdReturn" | awk '{print $3}')
    eval_avg=$(echo "$output" | grep "Eval_AverageReturn" | awk '{print $3}')
    eval_std=$(echo "$output" | grep "Eval_StdReturn" | awk '{print $3}')
    # initial_avg=$(echo "$output" | grep "Initial_DataCollection_AverageReturn" | awk '{print $3}')

    # Calculate percentage
    percentage=$(awk "BEGIN {printf \"%.3f\", ($eval_avg / $train_avg) * 100}")
    
    # Print agent name with result
    echo "${agent} eval/train: ${percentage}%"
    # Print Eval avg/std
    echo "Eval: ${eval_avg}/${eval_std}"
    # Print Train avg/std
    echo "Train: ${train_avg}/${train_std}"
    echo "---------------------------------"
}

# Example usage with different agents
for agent in Ant HalfCheetah Hopper Humanoid Walker2d; do
    run_agent "$agent"
done