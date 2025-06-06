import matplotlib.pyplot as plt
import numpy as np
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

plt.style.use('_mpl-gallery')

# make data:
np.random.seed(10)
colors = ["b","orange","g","r","purple","brown","pink","gray"]

names = ["Multi-Plate", "Particle Spring", "Multi-Body", "Finite Element"]
data_mp = [6, 12, 18, 33]
data_ps = [15, 15, 39, 111, 1122]
data_mb = [400]
data_fe = [240, 1332, 30000, 30000]
data = [data_mp, data_ps, data_mb, data_fe]

fig, ax = plt.subplots(figsize=(7, 4))

positions = np.arange(len(names)) + 1  # [1, 2, 3, 4]

for i, (pos, d) in enumerate(zip(positions, data)):
    whisker_low = np.min(d)
    whisker_high = np.max(d)
    ax.hlines(pos, whisker_low, whisker_high, color=colors[0], linewidth=2)
    ax.vlines([whisker_low, whisker_high], pos-0.2, pos+0.2, color=colors[0], linewidth=2)
    # Optionally, plot the median
    median = np.median(d)

ax.set_yticks(positions)
ax.set_yticklabels(names)
ax.grid(which='both')
ax.set_xlabel("Degrees of Freedom")
ax.set_ylim(0.5, len(names) + 0.5)
ax.set_xscale('log')

fig.tight_layout()
fig.savefig(os.path.join(script_dir,"DOF_comparison.svg"))
