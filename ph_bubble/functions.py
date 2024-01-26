import numpy as np
import matplotlib.pyplot as plt

"""method for sampling hexagon"""


class Sampling_kx_ky:

    def __equation_1(self, x):
        return - 1 / np.sqrt(3) * x + 4 * np.pi / 3

    def __equation_2(self, x):
        return 1 / np.sqrt(3) * x + 4 * np.pi / 3

    def sampling(self, N):
        """

        :type N: int
        """
        a = 2 * np.pi / np.sqrt(3)
        b = np.pi * 4 / 3
        random_points = np.zeros((N, 2))
        for i in range(N):
            random_x = np.random.uniform(-a, a)
            random_y = np.random.uniform(-b, b)
            random_point = np.array([random_x, random_y])
            while not (abs(random_y) <= abs(self.__equation_1(random_x)) and abs(random_y) <= abs(
                    self.__equation_2(random_x))):
                random_x = np.random.uniform(-a, a)
                random_y = np.random.uniform(-b, b)
                random_point = np.array([random_x, random_y])
            random_points[i] = random_point
        return random_points

    def sample_cut(self, n_cut):
        """n cut is number of points in every direction"""
        cut_x = np.linspace(0, 2 * np.pi / np.sqrt(3), n_cut)
        cut_x = np.append(cut_x, np.linspace(2 * np.pi / np.sqrt(3), 2 * np.pi / np.sqrt(3), n_cut))
        cut_x = np.append(cut_x, np.linspace(2 * np.pi / np.sqrt(3), 0, n_cut))
        cut_y = np.linspace(0, 0, n_cut)
        cut_y = np.append(cut_y, np.linspace(0, 2 * np.pi / 3, n_cut))
        cut_y = np.append(cut_y, np.linspace(2 * np.pi / 3, 0, n_cut))
        return cut_x, cut_y


# calculate bubble objects
class Bubble:

    def __init__(self, t, beta, tp=0):
        """parameters, t prime set to zero but can change when create object of class Bubble"""
        self.t = t
        self.tp = tp
        self.beta = beta

    def get_dispersion_triangular(self, kx, ky, mu):
        epsilon = (- self.t * (2 * np.cos(ky) + 4 * np.cos(ky / 2) * np.cos(kx * np.sqrt(3) / 2)) -
                   self.tp * 2 * (np.cos(np.sqrt(3) * kx) + 2 * np.cos(np.sqrt(3) * kx / 2) * np.cos(3 * ky / 2)) - mu)
        return epsilon

    def get_dispersion_triangular_cut(self, N_cut, mu):
        sample = Sampling_kx_ky()
        cut_x, cut_y = sample.sample_cut(N_cut)
        epsilon = np.vectorize(self.get_dispersion_triangular)(cut_x, cut_y, mu)
        return epsilon

    def fermi(self, kx, ky, mu):
        return 1 / (np.exp(self.beta * self.get_dispersion_triangular(kx, ky, mu)) + 1)

    def lindhard(self, kx, ky, qx, qy, Omega, mu):
        return -(self.fermi(kx, ky, mu) - self.fermi(kx + qx, ky + qy, mu)) / \
            (self.get_dispersion_triangular(kx, ky, mu) - self.get_dispersion_triangular(kx + qx, ky + qy, mu) + Omega)

    def spectral_func(self, kx, ky, omega, mu):
        return np.imag(1 / (omega - self.get_dispersion_triangular(kx, ky, mu) - 0.01j))

    def integrate_spectral(self, omega, N_samples, mu):
        sample = Sampling_kx_ky()
        random_k = sample.sampling(N_samples)
        integral = sum(self.spectral_func(random_k[:, 0], random_k[:, 1], omega, mu)) / N_samples
        return integral

    def integrate_lindhard(self, qx, qy, Omega, N_samples, mu):
        sample = Sampling_kx_ky()
        random_k = sample.sampling(N_samples)
        integral = sum(self.lindhard(random_k[:, 0], random_k[:, 1], qx, qy, Omega, mu)) / N_samples
        return integral

    def integrate_lindhard_cut(self, Omega, N_cut, N_samples, mu):
        sample = Sampling_kx_ky()
        cut_x, cut_y = sample.sample_cut(N_cut)
        integral_cut = np.vectorize(self.integrate_lindhard)(cut_x, cut_y, Omega, N_samples, mu)
        return integral_cut

    def integrate_lindhard_mu(self, qx, qy, Omega, N_samples, mu: np.array):
        integral_mu = np.vectorize(self.integrate_lindhard)(qx, qy, Omega, N_samples, mu)
        return integral_mu
