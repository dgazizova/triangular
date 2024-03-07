"""make density of states for the different omega values and plot fermi surface when omega = 0"""
import numpy as np
from matplotlib import pyplot as plt, tri

from ph_bubble.functions import Bubble, Sampling_kx_ky


spectral = Bubble(t=1, beta=5, tp=0)
sample = Sampling_kx_ky()
"""fermi surface in k space using triangular grid"""
# number of points for sampling in the 2d k space
N_samples = 100000
# dos = np.zeros(len(omega))



N = 20
uniform = sample.uniform_sample_q_space(n_vector=16)
fermi_surf = spectral.spectral_surface(omega=0, N_vector=16, mu=2)
print(len(uniform[0]))

kx = uniform[0]
ky = uniform[1]
for i, j in zip(kx, ky):
    print(i, j)


triangulation = tri.Triangulation(kx, ky)

# Create the figure and axes
fig, ax = plt.subplots()

# Plot the filled contour plot
contour = ax.tricontourf(triangulation, fermi_surf, levels=14)

# Add a colorbar
fig.colorbar(contour)

# Show the plot
plt.show()
