import numpy as np
import matplotlib.pyplot as plt



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

    def uniform_sample(self, n_vector):
        """function to uniformly sample hexagon using triangular grid"""
        #2 vectors of triangular lattice
        k1 = [2 * np.pi / np.sqrt(3), 2 * np.pi / 3]
        k2 = [0, 4 * np.pi / 3]
        uniform_points = [[], []]
        k1 = np.array(k1) / n_vector
        k2 = np.array(k2) / n_vector
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

    def uniform_sample_triangle(self, n_vector):
        """function to uniformly sample only one triangle of hexagon using triangular grid"""
        #2 vectors of triangular lattice
        k1 = [2 * np.pi / np.sqrt(3), 2 * np.pi / 3]
        k2 = [0, 4 * np.pi / 3]
        uniform_points = [[], []]
        k1 = np.array(k1) / n_vector
        k2 = np.array(k2) / n_vector
        for i in range(0, n_vector + 1):
            for j in range(0, n_vector + 1):
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
        return np.imag(1 / (omega - self.get_dispersion_triangular(kx, ky, mu) - 0.1j)) * 1/np.pi

    def integrate_spectral(self, omega, N_samples, mu):
        sample = Sampling_kx_ky()
        random_k = sample.sampling(N_samples)
        integral = sum(self.spectral_func(random_k[:, 0], random_k[:, 1], omega, mu)) / N_samples
        return integral

    def integrate_spectral_uniform(self, omega, N_vector, mu):
        sample = Sampling_kx_ky()
        uniform_k = sample.uniform_sample(n_vector=N_vector)
        N_samples = len(uniform_k[0])
        integral = sum(self.spectral_func(uniform_k[0], uniform_k[1], omega, mu)) / N_samples
        return integral, N_samples

    def spectral_surface(self, omega, N_vector, mu):
        sample = Sampling_kx_ky()
        uniform_k = sample.uniform_sample(n_vector=N_vector)
        # N_samples = len(uniform_k[0])
        spectral = np.vectorize(self.spectral_func)(uniform_k[0], uniform_k[1], omega, mu)
        return spectral


    def integrate_spectral_uniform_triangle(self, omega, N_vector, mu):
        sample = Sampling_kx_ky()
        uniform_k = sample.uniform_sample_triangle(n_vector=N_vector)
        N_samples = len(uniform_k[0])
        integral = sum(self.spectral_func(uniform_k[0], uniform_k[1], omega, mu)) / N_samples
        return integral, N_samples

    def integrate_spectral_uniform_cut(self, omega, N_cut, mu):
        sample = Sampling_kx_ky()
        cut_x, cut_y = sample.sample_cut(N_cut)
        N_samples = len(cut_x)
        integral = sum(self.spectral_func(cut_x, cut_y, omega, mu)) / N_samples
        return integral, N_samples


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

    def integrate_lindhard_meshgrid(self, qx, qy, Omega, N_samples, mu):
        integral = np.vectorize(self.integrate_lindhard)(qx, qy, Omega, N_samples, mu)
        return integral