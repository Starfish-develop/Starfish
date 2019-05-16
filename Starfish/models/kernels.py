import numpy as np
from typing import List

from Starfish import constants as C


def k_global_matrix(wave: np.ndarray, amplitude: float, lengthscale: float) -> np.ndarray:
    """
    A matern-3/2 kernel where the metric is defined as the velocity separation of the wavelengths.

    Parameters
    ----------
    wave : numpy.ndarray
        The wavelength grid
    amplitude : float
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
    kernel = amplitude * (1 + np.sqrt(3) * r/lengthscale) * \
        np.exp(-np.sqrt(3) * r / lengthscale)

    return kernel


def k_local_matrix(wave: np.ndarray, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    """
    A local Gaussian kernel. In general, the kernel has density like

    .. math::
        K(\\lambda | A, \\mu, \\sigma) = A \\exp\\left[-\\frac12 \\frac{\\left(\\lambda - \\mu\\right)^2}{\\sigma^2} \\right]

    Parameters
    ----------
    wave : numpy.ndarray
        The wavelength grid
    amplitude : float
        The amplitudes of the Gaussian
    mu : float
        The means of the Gaussian
    sigma : float
        The standard deviations of the Gaussian

    Returns
    -------
    cov : numpy.ndarray
        The sum of each Gaussian kernel, or the local covariance kernel
    """
    met = C.c_kms / mu * np.abs(wave - mu)
    x, y = np.meshgrid(met, met)
    r_tap = np.max([x, y], axis=0)
    r2 = x**2 + y**2
    r0 = 4 * sigma
    taper = 0.5 + 0.5 * np.cos(np.pi * r_tap / r0)
    taper[r_tap > r0] = 0
    kernel = taper * amplitude * np.exp(-0.5 * r2 / sigma**2)
    return kernel
