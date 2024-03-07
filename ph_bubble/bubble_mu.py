"""bubble diagram calculation for k space point calculation with different mu values"""
import numpy as np
import matplotlib.pyplot as plt
from functions import Sampling_kx_ky, Bubble

"""generate bubble for different mu points and qx and qy is corner points (may change the coordinate of the point)"""
#corner points of the triangular cut
n_sampling = 100000
Gamma = [0, 0]
K = [2 * np.pi/ np.sqrt(3), 2 * np.pi/3]
M = [2 * np.pi/ np.sqrt(3), 0]
kf2 = [3*np.pi/(2*np.sqrt(3)), np.pi/2]
kf3 = [np.pi/np.sqrt(3), np.pi]

# mu should be numpy array
mu = np.linspace(-2, 2.5, 30)
t = 1
tp = 0
bubble = Bubble(t=t, beta=5, tp=tp)
bubble_res_mu = bubble.integrate_lindhard_mu(qx=kf2[0], qy=kf2[1], Omega=0, N_samples=n_sampling, mu=mu)

plt.plot(mu, bubble_res_mu, label=f't={t}, tp={tp}, kf2 point')
plt.legend()
# plt.grid()
plt.show()
