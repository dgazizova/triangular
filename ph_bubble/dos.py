"""make density of states for the different omega values and plot fermi surface when omega = 0"""
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
spectral = Bubble(t=1, beta=5)
fermi_surf = spectral.spectral_func(kx_fs, ky_fs, 0, 0)


"""make dos for the different omega, sum over k space"""
# number of points for sampling in the 2d k space
N_samples = 100000
omega = np.linspace(-6, 4, 100)
dos = np.zeros(len(omega))
for i in range(len(omega)):
    dos[i] = spectral.integrate_spectral(omega[i], N_samples, 0)





"""corner points, 6 nearest neighbors"""
corner_x = [0, 2 / np.sqrt(3) * np.pi, 2 / np.sqrt(3) * np.pi, 0, -2 / np.sqrt(3) * np.pi, -2 / np.sqrt(3) * np.pi, 0]
corner_y = [4 * np.pi / 3, 2 * np.pi / 3, -2 * np.pi / 3, -4 * np.pi / 3, -2 * np.pi / 3, 2 * np.pi / 3, 4 * np.pi / 3]


"""plot fermi surface with the first BZ white lines"""
plt.pcolormesh(kx_fs, ky_fs, fermi_surf)
plt.plot(corner_x, corner_y, '--', color="white")
plt.text(np.pi / np.sqrt(3), np.pi/2, ".",  fontsize=16, color='red')
plt.colorbar()
plt.xlabel('kx')
plt.ylabel('ky')
plt.title('Fermi surface')
plt.show()

#plot dos for different omega points
plt.plot(omega, dos, marker='.')
plt.show()