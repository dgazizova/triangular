"""sampling points inside triangular lattice"""
import numpy as np
import matplotlib.pyplot as plt
from square_func import sampling, lindhard


qx = np.linspace(0.01, np.pi, 10)
qy = np.linspace(0.01, 0.01, 10)


"""bubble diagram through the cut"""
n_sampling = 100000
mu = np.linspace(-3, 3, 20)
bubble_res_mu = np.zeros(len(qx))
for i in range(len(qx)):
    random_points = sampling(n_sampling)
    bubble_res_mu[i] = sum(lindhard(beta=5, t=1, kx=random_points[:, 0], ky=random_points[:, 1], qx=qx[i], qy=qy[i], Omega=3,
                                    mu=0)) / (n_sampling)



plt.plot(qx, bubble_res_mu)
plt.show()

plt.plot(qx, -bubble_res_mu*bubble_res_mu)
plt.show()

spin = bubble_res_mu - -bubble_res_mu*bubble_res_mu
charge = bubble_res_mu - bubble_res_mu*bubble_res_mu
plt.plot(qx, spin, label=spin)
plt.plot(qx, charge, label=charge)
plt.legend()
plt.show()