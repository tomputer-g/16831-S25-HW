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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, log_dir in enumerate(log_dirs[:3]):
        returns = get_eval_average_returns(log_dir)
        if returns is not None:
            ax1.plot(returns, label=legends[0][i])
    
    for i, log_dir in enumerate(log_dirs[3:]):
        returns = get_eval_average_returns(log_dir)
        if returns is not None:
            ax2.plot(returns, label=legends[1][i])
    
    ax1.set_title('Small Batch')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Eval Average Return')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Large Batch')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Eval Average Return')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Six log directories (you can replace these with your actual log directories)
    log_dir_root = "/home/tomg/src/16831-S25-HW/hw2/data_submit/ex1/"
    log_dirs = [
        log_dir_root + "q1_sb_no_rtg_dsa_CartPole-v0_20-02-2025_14-45-37",
        log_dir_root + "q1_sb_rtg_dsa_CartPole-v0_20-02-2025_14-46-45",
        log_dir_root + "q1_sb_rtg_na_CartPole-v0_20-02-2025_14-47-52",
        log_dir_root + "q1_lb_no_rtg_dsa_CartPole-v0_20-02-2025_14-48-59",
        log_dir_root + "q1_lb_rtg_dsa_CartPole-v0_20-02-2025_14-52-58",
        log_dir_root + "q1_lb_rtg_na_CartPole-v0_20-02-2025_14-56-54"
    ]

    # Configurable legends for each subplot
    legends = [
        ["no_rtg_dsa", "rtg_dsa", "rtg_na"],
        ["no_rtg_dsa", "rtg_dsa", "rtg_na"]
    ]

    plot_returns(log_dirs, legends)
