import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYWORD = "Train_AverageReturn"
def get_eval_average_returns(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    if KEYWORD not in event_acc.Tags()['scalars']:
        print(f"Error: KEYWORD not found in the log file: {log_dir}")
        return None

    eval_average_returns = event_acc.Scalars(KEYWORD)

    if not eval_average_returns:
        print(f"Error: No KEYWORD events found in the log file: {log_dir}")
        return None

    return_values = np.array([event.value for event in eval_average_returns])
    return return_values

def plot_returns(log_dirs):
    plt.figure(figsize=(12,8))
    
    dqn_returns = None
    for i, log_dir in enumerate(log_dirs[:3]):
        if dqn_returns is None:
            dqn_returns = get_eval_average_returns(log_dir)
        else:
            dqn_returns = dqn_returns + get_eval_average_returns(log_dir)
    
    ddqn_returns = None
    for i, log_dir in enumerate(log_dirs[3:]):
        if ddqn_returns is None:
            ddqn_returns = get_eval_average_returns(log_dir)
        else:
            ddqn_returns = ddqn_returns + get_eval_average_returns(log_dir)
    dqn_returns = dqn_returns * (1/3)
    ddqn_returns = ddqn_returns * (1/3)
    
    plt.plot(dqn_returns, label="DQN Average Returns")
    plt.plot(ddqn_returns, label="DDQN Average Returns")
    plt.title("LunarLander v3 Average Returns with DQN vs. DDQN averaged across three runs")
    plt.xlabel('Steps * 10k')
    plt.ylabel('Average Return')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    print("Saving figure...")
    plt.savefig("fig_out/hw3_q1.png",dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # Six log directories (you can replace these with your actual log directories)
    log_dir_root = "/home/tomg/src/16831-S25-HW/hw3/data/"
    log_dirs = [
        log_dir_root + "q1_dqn_1_LunarLander-v3_05-03-2025_21-08-46",
        log_dir_root + "q1_dqn_2_LunarLander-v3_05-03-2025_21-15-38",
        log_dir_root + "q1_dqn_3_LunarLander-v3_05-03-2025_21-23-05",
        log_dir_root + "q1_doubledqn_1_LunarLander-v3_05-03-2025_21-30-10",
        log_dir_root + "q1_doubledqn_2_LunarLander-v3_05-03-2025_21-37-44",
        log_dir_root + "q1_doubledqn_3_LunarLander-v3_05-03-2025_21-45-52"
    ]

    plot_returns(log_dirs)
