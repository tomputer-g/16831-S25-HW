import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_eval_average_returns(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    if 'Eval_AverageReturn' not in event_acc.Tags()['scalars']:
        print(f"Error: 'Eval_AverageReturn' not found in the log file: {log_dir}")
        return None

    eval_average_returns = event_acc.Scalars('Eval_AverageReturn')

    if not eval_average_returns:
        print(f"Error: No 'Eval_AverageReturn' events found in the log file: {log_dir}")
        return None

    return_values = np.array([event.value for event in eval_average_returns])
    return return_values

def plot_returns(log_dirs):
    plt.figure(figsize=(12, 8))
    
    for log_dir in log_dirs:
        returns = get_eval_average_returns(log_dir)
        if returns is not None:
            # Extract batch size and learning rate from the log directory name
            parts = log_dir.split('_')
            batch_size = parts[4][1:]  # Remove 'b' prefix
            learning_rate = parts[5][2:]  # Remove 'r' prefix
            label = f"b={batch_size}, r={learning_rate}"
            plt.plot(returns, label=label)
    
    plt.title('Eval Average Return for Different Configurations')
    plt.xlabel('Steps')
    plt.ylabel('Eval Average Return')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    log_dir_root = "/home/tomg/src/16831-S25-HW/hw2/data_submit/ex4_search/"
    log_dirs = [
        log_dir_root + "q4_search_b10000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_20-02-2025_15-38-53",
        log_dir_root + "q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_20-02-2025_15-48-28",
        log_dir_root + "q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_20-02-2025_15-57-54",
        log_dir_root + "q4_search_b30000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_20-02-2025_16-07-14",
        log_dir_root + "q4_search_b30000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_20-02-2025_16-34-21",
        log_dir_root + "q4_search_b30000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_20-02-2025_17-01-23",
        log_dir_root + "q4_search_b50000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_20-02-2025_17-28-14",
        log_dir_root + "q4_search_b50000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_20-02-2025_18-13-52",
        log_dir_root + "q4_search_b50000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_20-02-2025_18-58-57",
    ]

    plot_returns(log_dirs)
