python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
	--train_batch_size 500 \
  	--num_agent_train_steps_per_iter 2000 \
 	--video_log_freq -1 \
    --n_layers 2
