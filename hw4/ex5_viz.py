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

def plot_returns(log_dirs, legends):
    plt.figure(figsize=(12, 8))
    
    for idx in range(len(log_dirs)):
        log_dir = log_dirs[idx]
        returns = get_eval_average_returns(log_dir)
        assert returns is not None
        plt.plot(returns, label=legends[idx])
    plt.title("MBRL with higher CEM iterations significantly outperforms lower iteration CEM or Random-shooting")
    plt.xlabel('Steps')
    plt.ylabel('Eval Average Return')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_out/hw4_q5.png",dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    log_dir_root = "/home/tomg/src/16831-S25-HW/hw4/rob831/data/"
    log_dirs = [
        log_dir_root + "hw4_q5_cheetah_random_cheetah-hw4_part1-v0_25-03-2025_21-07-17",
        log_dir_root + "hw4_q5_cheetah_cem_2_cheetah-hw4_part1-v0_25-03-2025_20-54-04",
        log_dir_root + "hw4_q5_cheetah_cem_4_cheetah-hw4_part1-v0_25-03-2025_21-19-16",
    ]
    legends = ["Random", "CEM 2", "CEM 4"]




    plot_returns(log_dirs, legends)
