import numpy as np
from typing import List

from Starfish import constants as C


def global_covariance_matrix(
    wave: np.ndarray, temp: float, amplitude: float, lengthscale: float
) -> np.ndarray:
    """
    A matern-3/2 kernel scaled by the planck function where the metric is defined as the velocity separation of the wavelengths.

    Parameters
    ----------
    wave : numpy.ndarray
        The wavelength grid
    temp : float
        The temperature of the current spectrum
    amplitude : float
        The amplitude of the kernel
    lengthscale : float
        The lengthscale of the kernel

    Returns
    -------
    cov : numpy.ndarray
        The global covariance matrix
    """
    wx, wy = np.meshgrid(wave, wave)
    r = C.c_kms / 2 * np.abs((wx - wy) / (wx + wy))
    r0 = 6 * lengthscale

    # Calculate the kernel, being careful to stay in mask
    kernel = np.zeros((len(wx), len(wy)))
    mask = r <= r0
    taper = 0.5 + 0.5 * np.cos(np.pi * r[mask] / r0)
    kernel[mask] = (
        taper
        * amplitude
        * (1 + np.sqrt(3) * r[mask] / lengthscale)
        * np.exp(-np.sqrt(3) * r[mask] / lengthscale)
    )
    planck = wave ** 5 * (np.exp(C.hc_k / wave / temp) - 1)
    return kernel / planck


def local_covariance_matrix(
    wave: np.ndarray, amplitude: float, mu: float, sigma: float
) -> np.ndarray:
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
    # Set up the metric and mesh grid
    met = C.c_kms / mu * np.abs(wave - mu)
    x, y = np.meshgrid(met, met)
    r_tap = np.max([x, y], axis=0)
    r2 = x ** 2 + y ** 2
    r0 = 4 * sigma

    # Calculate the kernel. Use masking to keep sparse-ish calculations
    kernel = np.zeros((len(x), len(y)))
    mask = r_tap <= r0
    taper = 0.5 + 0.5 * np.cos(np.pi * r_tap[mask] / r0)
    kernel[mask] = taper * amplitude * np.exp(-0.5 * r2[mask] / sigma ** 2)
    return kernel
