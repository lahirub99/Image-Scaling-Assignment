import matplotlib.pyplot as plt

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

# Now you can work with each subplot using the 'axes' array
axes[0, 0].plot([1, 2, 3, 4], [1, 4, 9, 16])
axes[0, 0].set_title('Subplot 1 - Plot')

axes[0, 1].scatter([1, 2, 3, 4], [1, 4, 9, 16])
axes[0, 1].set_title('Subplot 2 - Scatter')

axes[1, 0].bar([1, 2, 3, 4], [1, 4, 9, 16])
axes[1, 0].set_title('Subplot 3 - Bar')

axes[1, 1].hist([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
axes[1, 1].set_title('Subplot 4 - Histogram')

# Adjust layout to prevent overlapping titles and labels
plt.tight_layout()

# Show the figure
plt.show()
