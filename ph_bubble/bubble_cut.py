"""bubble diagram calculation for the cut G-M-K-G """
import numpy as np
import matplotlib.pyplot as plt
from functions import Sampling_kx_ky, Bubble

#number of samples throgh cut
N = 20 # N in every direction since its triangular cut N*3 total number of points

# if you want to generate this cut uncomment bellow
# sample = Sampling_kx_ky
# cut_x, cut_y = sample.sample_cut(N)

"""bubble diagram through the cut"""
n_sampling = 100000 # number of sampling of kx and ky in integration
bubble = Bubble(t=1, beta=5, tp=0)
#integral for points in cut just give it number of points in every direction of the cut
bubble_res_mu0 = bubble.integrate_lindhard_cut(Omega=0, N_cut=N, N_samples=n_sampling, mu=0)
print(len(bubble_res_mu0))


# bubble2 = Bubble(t=1, beta=100)
# bubble_res_mu2 = bubble2.integrate_lindhard_cut(Omega=0, N_cut=N, N_samples=n_sampling, mu=2)

print(bubble_res_mu0[0], bubble_res_mu0[57])
x = np.arange(N*3 - 2)
plt.plot(x, bubble_res_mu0, marker=".", label="mu=0")
# plt.plot(x, bubble_res_mu2, marker='.', label='mu=2')
plt.xticks([0, 18, 37, 57], ['G', 'M', 'K', 'G'])
plt.xlabel("Cut")
plt.grid()
plt.legend()
plt.show()


#when need to save data
# stacked_arrays = np.column_stack((bubble_res_mu0, bubble_res_mu2))
# np.savetxt("bubble_mu0_2.txt", stacked_arrays, delimiter=' ')










