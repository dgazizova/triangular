"""bubble diagram in q space, plot in the using square grid and 3d view of the function"""
import numpy as np
import matplotlib.pyplot as plt
from functions import Sampling_kx_ky, Bubble

n_sampling = 10000
"""corner points, 6 nearest neighbors"""
corner_x = [0, 2 / np.sqrt(3) * np.pi, 2 / np.sqrt(3) * np.pi, 0, -2 / np.sqrt(3) * np.pi, -2 / np.sqrt(3) * np.pi, 0]
corner_y = [4 * np.pi / 3, 2 * np.pi / 3, -2 * np.pi / 3, -4 * np.pi / 3, -2 * np.pi / 3, 2 * np.pi / 3, 4 * np.pi / 3]


bubble = Bubble(t=1, beta=5, tp=0)
qx = np.linspace(-2 * np.pi / np.sqrt(3), 2 * np.pi / np.sqrt(3), 30)
qy = np.linspace(-4 * np.pi / 3, 4 * np.pi / 3, 30)
qx, qy = np.meshgrid(qx, qy)
bubble_res_mu0 = bubble.integrate_lindhard_meshgrid(qx=qx.flatten(), qy=qy.flatten(), Omega=0, N_samples=n_sampling,
                                                    mu=2)

bubble_res_mu0 = bubble_res_mu0.reshape(qx.shape)
plt.figure()
plt.contourf(qx, qy, bubble_res_mu0)
plt.plot(corner_x, corner_y, marker=".", color="red", label="hexagon")
plt.colorbar()  # To add a colorbar
plt.title("Contour Plot")
plt.xlabel("qx")
plt.ylabel("qy")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(qx, qy, bubble_res_mu0)

ax.set_xlabel('qx')
ax.set_ylabel('qy')
ax.set_zlabel('bubble_res_mu0')

plt.show()
