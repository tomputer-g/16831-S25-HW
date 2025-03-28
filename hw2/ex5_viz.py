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
            lambda_v = parts[4][6:]  # Remove 'b' prefix
            label = f"lambda={lambda_v}"
            plt.plot(returns, label=label)
    plt.axhline(400, linestyle='--', color='r', label='Expected Avg. Return')
    plt.title('Hopper-v4 Eval Average Returns with various GAE Lambda values')
    plt.xlabel('Steps')
    plt.ylabel('Eval Average Return')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_out/hw2_ex5.png",dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    log_dir_root = "/home/tomg/src/16831-S25-HW/hw2/data_submit/ex5/"
    log_dirs = [
        log_dir_root + "q5_b2000_r0.001_lambda0_Hopper-v4_22-02-2025_16-53-02",
        log_dir_root + "q5_b2000_r0.001_lambda0.95_Hopper-v4_22-02-2025_17-00-46",
        log_dir_root + "q5_b2000_r0.001_lambda0.99_Hopper-v4_22-02-2025_17-08-40",
        log_dir_root + "q5_b2000_r0.001_lambda1_Hopper-v4_22-02-2025_17-16-27",
    ]

    plot_returns(log_dirs)
