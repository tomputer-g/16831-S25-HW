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

def plot_returns(log_dirs, legends, title, fname):
    plt.clf()
    plt.figure(figsize=(12, 8))
    
    for idx in range(len(log_dirs)):
        log_dir = log_dirs[idx]
        returns = get_eval_average_returns(log_dir)
        assert returns is not None
        plt.plot(returns, label=legends[idx])
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Eval Average Return')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname,dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    log_dir_root = "/home/tomg/src/16831-S25-HW/hw4/rob831/data/"
    log_dirs_ensemble = [
        log_dir_root + "hw4_q4_reacher_ensemble1_reacher-hw4_part1-v0_25-03-2025_18-46-48",
        log_dir_root + "hw4_q4_reacher_ensemble3_reacher-hw4_part1-v0_25-03-2025_18-51-56",
        log_dir_root + "hw4_q4_reacher_ensemble5_reacher-hw4_part1-v0_25-03-2025_19-02-30",
    ]
    legends_ensemble = ["Ensemble = 1", "Ensemble = 3", "Ensemble = 5"]
    fname_ensemble = "fig_out/hw4q4_ensemble.png"
    plot_returns(log_dirs_ensemble, legends_ensemble, "Larger ensembles produce lower variance and slightly higher returns", fname_ensemble)

    log_dirs_horizon = [
        log_dir_root + "hw4_q4_reacher_horizon5_reacher-hw4_part1-v0_25-03-2025_17-42-58",
        log_dir_root + "hw4_q4_reacher_horizon15_reacher-hw4_part1-v0_25-03-2025_17-49-08",
        log_dir_root + "hw4_q4_reacher_horizon30_reacher-hw4_part1-v0_25-03-2025_18-01-09",
    ]
    legends_horizon = ["Horizon = 5", "Horizon = 15", "Horizon = 30"]
    fname_horizon = "fig_out/hw4q4_horizon.png"
    plot_returns(log_dirs_horizon, legends_horizon, "Long planning horizons reduce model performance compared to shorter planning horizon", fname_horizon)

    log_dirs_numseq = [
        log_dir_root + "hw4_q4_reacher_numseq100_reacher-hw4_part1-v0_25-03-2025_18-24-37",
        log_dir_root + "hw4_q4_reacher_numseq1000_reacher-hw4_part1-v0_25-03-2025_18-36-07",
    ]
    legends_numseq = ["Num. seq = 100", "Num. seq = 1000"]
    fname_numseq = "fig_out/hw4q4_numseq.png"
    plot_returns(log_dirs_numseq, legends_numseq, "Higher number of candidate action sequences slightly increase model performance", fname_numseq)