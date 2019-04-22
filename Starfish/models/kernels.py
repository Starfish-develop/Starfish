import numpy as np

from Starfish import constants as C


@np.vectorize
def k_local(x0, x1, a, mu, sigma):
    r0 = 4.0 * sigma  # spot where kernel goes to 0
    rx0 = C.c_kms / mu * np.abs(x0 - mu)
    rx1 = C.c_kms / mu * np.abs(x1 - mu)
    r_tap = rx0 if rx0 > rx1 else rx1  # choose the larger distance

    if r_tap >= r0:
        return 0.
    else:
        taper = (0.5 + 0.5 * np.cos(np.pi * r_tap / r0))
        return taper * a ** 2 * np.exp(-0.5 * C.c_kms ** 2 / mu ** 2 * ((x0 - mu) ** 2 + (x1 - mu) ** 2) / sigma ** 2)


def k_local_func(x0i, x1i, x0v=None, x1v=None, a=None, mu=None, sigma=None):
    x0 = x0v[x0i]
    x1 = x1v[x1i]
    return k_local(x0=x0, x1=x1, a=a, mu=mu, sigma=sigma)


def Poisson_matrix(wl, sigma):
    '''
    Sigma can be an array or a single float.
    '''
    N = len(wl)
    matrix = sigma ** 2 * np.eye(N)
    return matrix


def k_global_matrix(wave, a, l):
    out = np.zeros((len(wave), len(wave)))
    for i, w1 in enumerate(wave):
        r0 = 6 * l
        r = C.c_kms / 2 * np.abs((wave - w1) / (wave + w1))

        taper = 1/2 + 1/2 * np.cos(np.pi * r / r0)
        matern = a**2 * (1 + np.sqrt(3) * r/l) * np.exp(-np.sqrt(3) * r / l)
        out[i, r < r0] = (taper * matern)[r < r0]

    return out

def k_local_matrix(wl, a, mu, sigma):
    N = len(wl)
    matrix = np.fromfunction(k_local_func, (N, N), x0v=wl,
                             x1v=wl, a=a, mu=mu, sigma=sigma, dtype=np.int)
    return matrix
