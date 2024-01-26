from ph_bubble.functions import get_dispersion_triangular
import numpy as np


M = [2*np.pi/np.sqrt(3),0]
G = [np.pi/np.sqrt(3), np.pi]
res = get_dispersion_triangular(t=1, kx=G[0], ky=G[1], mu=2)
print(res)