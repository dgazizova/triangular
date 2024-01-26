"""code that shows random sampling of the hexagon and dispersion relation through the cut"""


import numpy as np
import matplotlib.pyplot as plt
from ph_bubble.functions import Sampling_kx_ky, Bubble

a = 2 * np.pi / np.sqrt(3)
b = np.pi * 4 / 3


"""randomly sampling of N points """
N = 100000
sample = Sampling_kx_ky()
points = sample.sampling(N=N)


"""corner points, 6 nearest neighbors"""
corner_x = [0, 2 / np.sqrt(3) * np.pi, 2 / np.sqrt(3) * np.pi, 0, -2 / np.sqrt(3) * np.pi, -2 / np.sqrt(3) * np.pi, 0]
corner_y = [4 * np.pi / 3, 2 * np.pi / 3, -2 * np.pi / 3, -4 * np.pi / 3, -2 * np.pi / 3, 2 * np.pi / 3, 4 * np.pi / 3]
"""cut points through the triangular"""
cut_points_x = [0, 2 * np.pi / np.sqrt(3), 2 * np.pi / np.sqrt(3), 0]
cut_points_y = [0, 0, 2 * np.pi / 3, 0]


"""fermi surface cut"""
n = 100
cut_x, cut_y = sample.sample_cut(n)


disp = Bubble(t=1, beta=5, tp=-0.3)
dispersion = disp.get_dispersion_triangular_cut(N_cut=n, mu=0)

"""plot sampling points with the cut lines and hexagon around"""
plt.plot(corner_x, corner_y, marker=".", color="red", label="hexagon")
plt.plot(cut_x, cut_y, marker=".", color="black", label="cut")
plt.scatter(points[:, 0], points[:, 1], label='Random points', marker='.')
plt.text(0, 0.2, r'$\Gamma$', fontsize=16, color='red')
plt.text(2 * np.pi / np.sqrt(3), 0.2, r"M",  fontsize=16, color='red')
plt.text(2 * np.pi / np.sqrt(3), 2 * np.pi / 3 + 0.2, "K", fontsize=16, color='red')
plt.xlim(-a-1, a+1)
plt.ylim(-b-1, b+1)
plt.title('2D Plot of Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()


"""plot dispersion relation through the cut"""
plt.plot(dispersion, label="dispersion")
plt.legend()
plt.show()