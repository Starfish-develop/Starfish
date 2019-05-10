import numpy as np

from Starfish import constants as C


def k_global_matrix(wave, amp, lengthscale):
    """
    A matern-3/2 kernel where the metric is defined as the velocity separation of the wavelengths.

    Parameters
    ----------
    wave : numpy.ndarray
        The wavelength grid
    amp : float
        The amplitude of the kernel
    lengthscale : float
        The lengthscale of the kernel

    Returns
    -------
    cov : numpy.ndarray
        The global covariance matrix
    """
    r0 = 6 * lengthscale
    wx, wy = np.meshgrid(wave, wave)
    r = C.c_kms / 2 * np.abs((wx - wy) / (wx + wy))
    taper = 0.5 + 0.5 * np.cos(np.pi * r / r0)
    taper[r > r0] = 0
    kernel = amp * (1 + np.sqrt(3) * r/lengthscale) * \
        np.exp(-np.sqrt(3) * r / lengthscale)

    return kernel


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
    covs = []
    for amp, mu, sig in zip(amps, mus, sigs):
        met = C.c_kms / mu * np.abs(wave - mu)
        x, y = np.meshgrid(met, met)
        r_tap = np.max([x, y], axis=0)
        r2 = x**2 + y**2
        r0 = 4 * sig
        taper = 0.5 + 0.5 * np.cos(np.pi * r_tap / r0)
        taper[r_tap > r0] = 0
        cov = taper * amp * np.exp(-0.5 * r2 / sig**2)
        covs.append(cov)

    return np.sum(covs, axis=0)
