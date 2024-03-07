# from ph_bubble.functions import get_dispersion_triangular
import numpy as np
import matplotlib.pyplot as plt

def uniform_sample(n_vector):
    """function to uniformly sample hexagon using triangular grid"""
    # 2 vectors of triangular lattice
    k1 = [np.sqrt(3) / 2, 1 / 2]
    k2 = [0, 1]
    uniform_points = [[], []]
    k1 = np.array(k1)
    k2 = np.array(k2)
    for i in range(-n_vector, n_vector + 1):
        for j in range(-n_vector, n_vector + 1):
            kx = k1[0] * i + k2[0] * j
            ky = k1[1] * i + k2[1] * j
            # to cut points outside of hexagon
            if i + j > n_vector or i + j < - n_vector:
                continue
            uniform_points[0].append(kx)
            uniform_points[1].append(ky)
    uniform_points[0] = np.array(uniform_points[0])
    uniform_points[1] = np.array(uniform_points[1])
    return uniform_points


uniform_points = uniform_sample(10)
plt.xlim(-10.1, 10.1)
plt.ylim(-10.1, 10.1)
plt.scatter(uniform_points[0], uniform_points[1], label='unifrom points', marker='.')
plt.show()
