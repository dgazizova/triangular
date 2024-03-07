"""uniform triangle grid plot and fermi surface on this grid"""
import numpy as np
from matplotlib import pyplot as plt, tri

from ph_bubble.functions import Bubble, Sampling_kx_ky

sample = Sampling_kx_ky()
"uniform sampling of triangular lattice"
n_vector = 15 # number of reciprocal vector separation
uniform_points = sample.uniform_sample_q_space(n_vector=n_vector)
print(len(uniform_points[0]))

"""plot picture of uniform sampling of the hexagon"""
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.scatter(uniform_points[0], uniform_points[1], label='unifrom points', marker='.')
plt.show()

spectral = Bubble(t=1, beta=5, tp=0)
"""fermi surface in k space using triangle grid"""
fermi_surf = spectral.spectral_surface(omega=0, N_vector=n_vector, mu=2)
kx = uniform_points[0]
ky = uniform_points[1]
triangulation = tri.Triangulation(kx, ky)

# Create the figure and axes
fig, ax = plt.subplots()

# Plot the filled contour plot
contour = ax.tricontourf(triangulation, fermi_surf, levels=14)

# Add a colorbar
fig.colorbar(contour)

# Show the plot
plt.show()
