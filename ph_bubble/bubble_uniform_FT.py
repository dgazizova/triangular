"""1. uniform sampling of triangular lattice along q reciprocal vectors
   2. calculation of bubble diagram for these points
   3. FT to the real space along real vectors"""
import numpy as np
from matplotlib import pyplot as plt, tri
from functions import Sampling_kx_ky, Bubble


"""Number of points for integration"""
N = 10000


"""uniform sampling of triangular lattice"""
sample = Sampling_kx_ky()
n_q_vector = 20
uniform_points = sample.uniform_sample_q_space(n_vector=n_q_vector)
uniform_points = np.array(uniform_points)
uniform_points = uniform_points + 0.0001
print(len(uniform_points[0]))

bubble = Bubble(t=1, beta=5, tp=0)
bubble_res = bubble.integrate_lindhard_meshgrid(qx=uniform_points[0], qy=uniform_points[1], Omega=0, N_samples=N,
                                                    mu=0)

"""plot bubble diagram results"""
# print(bubble_res)
triangulation = tri.Triangulation(uniform_points[0], uniform_points[1])

# Create the figure and axes
fig, ax = plt.subplots()

# Plot the filled contour plot
contour = ax.tricontourf(triangulation, bubble_res, levels=14)

# Add a colorbar
fig.colorbar(contour)

# Show the plot
plt.show()


"""FT part"""
n_r_vector = 4 # number of points of real space vectors
bubble_r_space_cut_1 = bubble.FT(n_q_vector=n_q_vector, data_q=bubble_res, n_r_vector=n_r_vector, cut=True, cut_type=1)
bubble_r_space_cut_2 = bubble.FT(n_q_vector=n_q_vector, data_q=bubble_res, n_r_vector=n_r_vector, cut=True, cut_type=2)

"""plot FT cuts along real space vectors"""
plt.subplot(1, 2, 1)
plt.title("cut along a1")
plt.plot(bubble_r_space_cut_1)

plt.subplot(1, 2, 2)
plt.title("cut along a2")
plt.plot(bubble_r_space_cut_2)
plt.show()

"""plot FT in real space"""
uniform_points_r = sample.uniform_sample_q_space(n_vector=n_r_vector)
bubble_r_space = bubble.FT(n_q_vector=n_q_vector, data_q=bubble_res, n_r_vector=n_r_vector)


# print(bubble_res)
triangulation = tri.Triangulation(uniform_points_r[0], uniform_points_r[1])

# Create the figure and axes
fig, ax = plt.subplots()

# Plot the filled contour plot
contour = ax.tricontourf(triangulation, bubble_r_space, levels=14)

# Add a colorbar
fig.colorbar(contour)

# Show the plot
plt.show()