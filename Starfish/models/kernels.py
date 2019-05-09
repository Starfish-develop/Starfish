import numpy as np

from Starfish import constants as C


def k_global_matrix(wave, a, l):
    """
    A matern-3/2 kernel where the metric is defined as the velocity separation of the wavelengths.

    Parameters
    ----------
    wave : numpy.ndarray
        The wavelength grid
    a : float
        The amplitude of the kernel
    l : float
        The lengthscale of the kernel

    Returns
    -------
    cov : numpy.ndarray
        The global covariance matrix
    """
    out = np.zeros((len(wave), len(wave)))
    r0 = 6 * l
    for i, w1 in enumerate(wave):
        r = C.c_kms / 2 * np.abs((wave - w1) / (wave + w1))
        taper = 1/2 + 1/2 * np.cos(np.pi * r / r0)
        matern = a * (1 + np.sqrt(3) * r/l) * np.exp(-np.sqrt(3) * r / l)
        out[i, r < r0] = (taper * matern)[r < r0]

    return out


def k_local_matrix(wave, amps, mus, sigs):
    """
    A local Gaussian kernel.

    Parameters
    ----------
    wave : numpy.ndarray
        The wavelength grid
    amps : array_like
        The amplitudes of each Gaussian
    mus : array_like
        The means of each Gaussian
    sigs : array_like
        The standard deviations of each Gaussian

    Returns
    -------
    cov : numpy.ndarray
        The sum of each Gaussian kernel, or the local covariance kernel
    """
    out = np.zeros((len(wave), len(wave)))
    for amp, mu, sig in zip(amps, mus, sigs):
        r0 = 4 * sig
        metric = C.c_kms / mu * np.abs(wave - mu)
        for i, val in enumerate(metric):
            row = np.tile(val, len(metric))
            r_tap = np.max([row, metric], axis=0)
            taper = (1/2 + 1/2 * np.cos(np.pi * r_tap / r0))
            guass = taper * amp * \
                np.exp(-0.5 * (row**2 + metric**2) / sig ** 2)
            out[i, r_tap < r0] += guass[r_tap < r0]
    return out
