# python rob831/scripts/run_hw1.py \
#     --expert_policy_file rob831/policies/experts/Ant.pkl \
#     --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
#     --do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
#  	  --video_log_freq -1
#Take in agent name as argument. 
agent=$1
agent_lower="${agent,,}"          # Convert to lowercase (ant)

echo "Running $agent..."
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/$agent.pkl \
    --env_name "$agent-v2" --exp_name "dagger_$agent_lower" --n_iter 10 \
    --do_dagger --expert_data rob831/expert_data/expert_data_$agent-v2.pkl \
	  --train_batch_size 100 \
  	--num_agent_train_steps_per_iter 1000 \
    --ep_len 1000 --eval_batch_size 5000 \
 	  --video_log_freq -1 \
    --n_layers 1 2>/dev/null 1>"dagger_${agent_lower}_out"
echo "Done"
# python hw2q2_dagger_processing.py
python hw2q2_processing.py "$agent"