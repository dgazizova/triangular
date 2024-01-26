import numpy as np
def sampling(N):
    a = np.pi
    b = np.pi
    random_points = np.zeros((N, 2))
    for i in range(N):
        random_x = np.random.uniform(-a, a)
        random_y = np.random.uniform(-b, b)
        random_point = np.array([random_x, random_y])
        random_points[i] = random_point
    return random_points

def get_dispersion_triangular(t, kx, ky, mu):
    epsilon = - t * 2 * (np.cos(kx) + np.cos(ky)) - mu
    return epsilon

def fermi(beta, t, kx, ky, mu):
    return 1 / (np.exp(beta * get_dispersion_triangular(t, kx, ky, mu)) + 1)


def lindhard(beta, t, kx, ky, qx, qy, Omega, mu):
    return (fermi(beta, t, kx, ky, mu) - fermi(beta, t, kx + qx, ky + qy, mu)) / \
           (-get_dispersion_triangular(t, kx, ky, mu) + get_dispersion_triangular(t, kx + qx, ky + qy, mu) + Omega)