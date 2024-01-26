"""sampling points inside triangular lattice and  RPA calculation of W for triangular"""
import numpy as np
import matplotlib.pyplot as plt
from functions import Sampling_kx_ky, Bubble

#number of samples throgh cut
N = 30
# if you want to generate this cut uncomment bellow
# sample = Sampling_kx_ky
# cut_x, cut_y = sample.sample_cut(N)

"""bubble diagram through the cut"""
n_sampling = 100000


bubble = Bubble(t=1, beta=5, tp=0)
bubble_res_mu0 = bubble.integrate_lindhard_cut(Omega=0, N_cut=N, N_samples=n_sampling, mu=0)


bubble2 = Bubble(t=1, beta=5)
bubble_res_mu2 = bubble2.integrate_lindhard_cut(Omega=0, N_cut=N, N_samples=n_sampling, mu=2)

K = [2 * np.pi/ np.sqrt(3), 2 * np.pi/3]
M = [2 * np.pi/ np.sqrt(3), 0]

# mu should be numpy array
mu = np.linspace(-2, 2, 50)
bubble3 = Bubble(t=1, beta=5, tp=-0.3)
bubble_res_mu3 = bubble3.integrate_lindhard_mu(qx=M[0], qy=M[1], Omega=0, N_samples=n_sampling, mu=mu)

plt.plot(mu, bubble_res_mu3, label='bubble')
plt.legend()
plt.show()


# stacked_arrays = np.column_stack((bubble_res_mu0, bubble_res_mu2))
# np.savetxt("bubble_mu0_2.txt", stacked_arrays, delimiter=' ')


plt.plot(bubble_res_mu0, marker='.', label="mu=0")
# plt.plot(bubble_res_mu2, marker='.', label='mu=2')
plt.legend()
plt.show()








