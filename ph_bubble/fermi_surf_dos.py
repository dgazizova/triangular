"""fermi surface and density of states for the different omega values"""
import numpy as np
from matplotlib import pyplot as plt

from ph_bubble.functions import Bubble

a = 2 * np.pi / np.sqrt(3)
b = np.pi * 4 / 3


fermi_surf_number = 1000
"""find full fermi surface, distribution over 1000**2 points"""
kx_fs = np.linspace(-b, b, fermi_surf_number)
ky_fs = np.linspace(-b, b, fermi_surf_number)
kx_fs, ky_fs = np.meshgrid(kx_fs, ky_fs)

"""lifshits transition at mu = 2 at van hof singularity """
mu = 2.2
spectral = Bubble(t=1, beta=5, tp=0.1)
fermi_surf = spectral.spectral_func(kx=kx_fs, ky=ky_fs, omega=0, mu=mu)

"""corner points, 6 nearest neighbors"""
corner_x = [0, 2 / np.sqrt(3) * np.pi, 2 / np.sqrt(3) * np.pi, 0, -2 / np.sqrt(3) * np.pi, -2 / np.sqrt(3) * np.pi, 0]
corner_y = [4 * np.pi / 3, 2 * np.pi / 3, -2 * np.pi / 3, -4 * np.pi / 3, -2 * np.pi / 3, 2 * np.pi / 3, 4 * np.pi / 3]

"""plot fermi surface with the first BZ white lines"""
plt.pcolormesh(kx_fs, ky_fs, fermi_surf)
plt.plot(corner_x, corner_y, '--', color="white")
# plt.plot(np.pi / np.sqrt(3), np.pi/2, "o", color='red')
plt.colorbar()
plt.xlabel('kx')
plt.ylabel('ky')
plt.title(f'mu = {mu}')
plt.show()


"""make dos for the different omega, sum over k space"""
# number of points for sampling in the 2d k space
N_samples = 100000
omega = np.linspace(-6, 4, 100)
dos = np.zeros(len(omega))
for i in range(len(omega)):
    # uncomment bellow for random sampling of k points in dos
    # dos[i] = spectral.integrate_spectral(omega[i], N_samples, 0)

    # integrate_spectral_uniform samples through hexagon and integrate_spectral_uniform_triangle sample through triangle
    dos[i], N_k_samples = spectral.integrate_spectral_uniform(omega=omega[i], N_vector=300, mu=0)
    # dos[i], N_v_samples = spectral.integrate_spectral_uniform_cut(omega=omega[i], N_cut=2000, mu=0)

print(N_k_samples)



"""plot dos for different omega values"""
plt.plot(omega, dos, marker='.')
plt.show()
