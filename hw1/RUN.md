# Reproduction of data

## Question 1

For Q1.2 (expert policy trajectories), run this helper script which prints the training mean/stds to command line directly:

```bash
./hw1q2_expert_policy_two_trajs.sh
```

For Q1.3 (behavior cloning results), run this helper script which prints the train and eval mean/stds, as well as the percentage of expert performance achieved, to command line directly:

```bash
./hw1q3_baseline_tasks.sh
```

For Q1.4 (hyperparameter), run the bash script which prints the evaluation mean/stds to command line, then modify the lists in hw1q4_graph_performance.py and execute it to produce the graph:

```bash
./hw1q4_hyperparams.sh
python hw1q4_graph_performance.py
```

## Question 2

For the DAgger results, run:

```bash
./hw2q2_dagger.sh <env_name>
# Example: ./hw2q2_dagger.sh Ant
```

This collects the results and produces a plot with the mean/stds. It uses the hw2q2_processing.py as part of the script.