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

def plot_returns(log_dirs, labels):
    plt.figure(figsize=(12, 8))
    
    for i,log_dir in enumerate(log_dirs):
        returns = get_eval_average_returns(log_dir)
        if returns is not None:
            # Extract batch size and learning rate from the log directory name
            
            plt.plot(returns, label=labels[i])
    
    plt.title('Eval Average Return for Different Configurations')
    plt.xlabel('Steps')
    plt.ylabel('Eval Average Return')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_out/hw2_ex4_comp.png",dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    log_dir_root = "/home/tomg/src/16831-S25-HW/hw2/data_submit/ex4_comp/"
    log_dirs = [
        log_dir_root + "q4_b30000_r0.02_HalfCheetah-v4_21-02-2025_21-11-58",
        log_dir_root + "q4_b30000_r0.02_nnbaseline_HalfCheetah-v4_21-02-2025_22-06-16",
        log_dir_root + "q4_b30000_r0.02_rtg_HalfCheetah-v4_21-02-2025_21-39-08",
        log_dir_root + "q4_b30000_r0.02_rtg_nnbaseline_HalfCheetah-v4_21-02-2025_22-33-34",
    ]

    labels = [
        "No RTG, No NNBaseline",
        "No RTG, NNBaseline",
        "RTG, No NNBaseline",
        "RTG, NNBaseline",
    ]

    plot_returns(log_dirs, labels=labels)
