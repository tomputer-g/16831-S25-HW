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
            plt.plot(returns)
    plt.axhline(120, linestyle='--', color='r', label='Expected Avg. Return')
    
    plt.title('LunarLanderContinuous-v2 Eval Average Returns')
    plt.xlabel('Steps')
    plt.ylabel('Eval Average Return')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.savefig("fig_out/hw2_ex3_lander.png",dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

    # print("Saving figure...")
    # plt.savefig("fig_out/hw2_ex3_lander.png",dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    log_dir_root = "/home/tomg/src/16831-S25-HW/hw2/data_submit/ex3/"
    log_dirs = [
        log_dir_root + "q3_b10000_r0.005_LunarLanderContinuous-v2_20-02-2025_14-42-00",
    ]

    plot_returns(log_dirs)
