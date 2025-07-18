import matplotlib.pyplot as plt
import numpy as np
import glob

# Get all relevant files in the subfolder
file_list = glob.glob("data/data*.txt")

# Create a plot
plt.figure()

# Loop through each file and plot the data
for i,file in enumerate(file_list):
    data = np.loadtxt(file, skiprows=1)  # skip the header line
    x, y = data[:, 0], data[:, 1]
    label = f"{3*(i+1)}x{3*(i+1)} tiles"  # just the filename without the path
    if file != "img\data3.2.txt":
        plt.plot(x, y, label=label)

# Customize the plot

plt.xscale('log')
plt.yscale('log')
plt.xlabel('physical error rate')
plt.ylabel('logical error rate')
plt.title('Floquet colour code')
plt.grid(True,ls='--')
plt.legend()
plt.savefig('img/flo_col_comparison.png', dpi=400, bbox_inches='tight')
# Show the plot
# plt.show()
