import numpy as np

import matplotlib.pyplot as plt

# Sample data
means = [378.89, 456.22, 1209.39, 1239.36, 1129.40, 1068.33, 1180.01, 1201.15, 936.58, 1281.00]

stds = [30.48, 283.15, 198.51, 150.37, 37.00, 69.00, 88.74, 170.47, 72.90, 139.31]

# Number of entries
n = len(means)

# X-axis labels
labels = [200*i for i in range(1,n+1)]

# Create figure and axis
fig, ax = plt.subplots()

# Plotting the bar graph with error bars
ax.bar(range(n), means, yerr=stds, capsize=5, tick_label=labels)

# Adding labels and title
ax.set_xlabel('Training steps per iteration')
ax.set_ylabel('Hopper BC Agent Performance')
ax.set_title('Hopper Agent Performance vs. Training Steps per Iteration')

# Show plot
plt.show()