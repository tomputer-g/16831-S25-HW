import sys
import re
import matplotlib.pyplot as plt

def extract_values(log_output):
    eval_average_returns = []
    eval_std_returns = []
    
    avg_pattern = r"Eval_AverageReturn : (\d+\.\d+)"
    std_pattern = r"Eval_StdReturn : (\d+\.\d+)"
    
    avg_matches = re.findall(avg_pattern, log_output)
    std_matches = re.findall(std_pattern, log_output)
    
    eval_average_returns = [float(value) for value in avg_matches]
    eval_std_returns = [float(value) for value in std_matches]
    
    return eval_average_returns, eval_std_returns

EXPERT_PERF = {"Ant": 4713.65, "Hopper": 3772.67}
BC_PERF = {"Ant": 4538.17, "Hopper": 1129.40}

def plot_results(env_name, eval_average_returns, eval_std_returns):
    iterations = range(len(eval_average_returns))
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(iterations, eval_average_returns, yerr=eval_std_returns, fmt='o-', capsize=5, label='DAgger')
    
    # Add horizontal lines
    assert env_name in EXPERT_PERF, f"Expert performance for {env_name} not found."
    expert_perf = EXPERT_PERF[env_name]
    bc_perf = BC_PERF[env_name]
    plt.axhline(y=bc_perf, color='r', linestyle='--', label='BC Returns')
    plt.axhline(y=expert_perf, color='b', linestyle='--', label='Expert Returns')
    
    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title(f'DAgger Performance on {env_name}')
    plt.legend()
    plt.grid(True)
    
    # plt.savefig(f'dagger_{env_name.lower()}_plot.png')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <lower_case_env_name> (e.g. python cooler_hw2q2.py hopper)")
        sys.exit(1)

    env_name = sys.argv[1]
    filename = f"dagger_{env_name.lower()}_out"

    try:
        with open(filename, 'r') as file:
            log_output = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    eval_average_returns, eval_std_returns = extract_values(log_output)

    plot_results(env_name, eval_average_returns, eval_std_returns)
