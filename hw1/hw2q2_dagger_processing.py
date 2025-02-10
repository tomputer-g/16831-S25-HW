import re
import numpy as np
import matplotlib.pyplot as plt

def extract_values(log_output):
    eval_average_returns = []
    eval_std_returns = []
    
    # Regular expressions to match the required values
    avg_pattern = r"Eval_AverageReturn : (\d+\.\d+)"
    std_pattern = r"Eval_StdReturn : (\d+\.\d+)"
    
    # Find all matches in the log output
    avg_matches = re.findall(avg_pattern, log_output)
    std_matches = re.findall(std_pattern, log_output)
    
    # Convert matched strings to floats and add to respective lists
    eval_average_returns = [float(value) for value in avg_matches]
    eval_std_returns = [float(value) for value in std_matches]
    
    return eval_average_returns, eval_std_returns

# Read the log output from the 'dagger_out' file
with open('dagger_out', 'r') as file:
    log_output = file.read()

# Extract the values
means, stds = extract_values(log_output)

# Number of entries
n = len(means)

# X-axis labels
labels = [i for i in range(1,n+1)]

# Create figure and axis
fig, ax = plt.subplots()

# Plotting the bar graph with error bars
ax.bar(range(n), means, yerr=stds, capsize=5, tick_label=labels)

# Add horizontal line at 5000
ax.axhline(y=3772, color='r', linestyle='--')
# Ant BC: 4713.65
# Hopper BC: 3772.67

# Adding labels and title
ax.set_xlabel('Number of DAgger Iterations')
ax.set_ylabel('Agent Evaluation Performance')
ax.set_title('DAgger Hopper Agent Performance vs. DAgger iterations')

# Show plot
plt.show()