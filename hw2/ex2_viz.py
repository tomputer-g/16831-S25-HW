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
            batch_size = parts[2][1:]  # Remove 'b' prefix
            learning_rate = parts[3][1:]  # Remove 'r' prefix
            label = f"b={batch_size}, r={learning_rate}"
            plt.plot(returns, label=label)
    
    plt.title('InvertedPendulum-v4 Eval Average Return for Different Configurations')
    plt.xlabel('Steps')
    plt.ylabel('Eval Average Return')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    log_dir_root = "/home/tomg/src/16831-S25-HW/hw2/data_submit/ex2/"
    log_dirs = [
        # log_dir_root + "q2_b10000_r0.005_InvertedPendulum-v4_22-02-2025_09-53-52",
        log_dir_root + "q2_b10000_r0.01_InvertedPendulum-v4_22-02-2025_10-02-57",
        # log_dir_root + "q2_b10000_r0.02_InvertedPendulum-v4_22-02-2025_10-11-57",
        # log_dir_root + "q2_b20000_r0.005_InvertedPendulum-v4_22-02-2025_10-21-01",
        log_dir_root + "q2_b20000_r0.01_InvertedPendulum-v4_22-02-2025_10-38-23",
        log_dir_root + "q2_b20000_r0.02_InvertedPendulum-v4_22-02-2025_10-55-39",
        # log_dir_root + "q2_b10000_r0.03_InvertedPendulum-v4_22-02-2025_16-00-48",
        # log_dir_root + "q2_b20000_r0.03_InvertedPendulum-v4_22-02-2025_16-30-37",
        # log_dir_root + "q2_b5000_r0.005_InvertedPendulum-v4_22-02-2025_09-39-08",
        # log_dir_root + "q2_b5000_r0.01_InvertedPendulum-v4_22-02-2025_09-44-06",
        # log_dir_root + "q2_b5000_r0.02_InvertedPendulum-v4_22-02-2025_09-49-04",
        # log_dir_root + "q2_b5000_r0.03_InvertedPendulum-v4_23-02-2025_11-12-03",
    ]

    plot_returns(log_dirs)
