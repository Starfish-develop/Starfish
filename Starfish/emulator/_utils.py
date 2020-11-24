import logging

import numpy as np
import scipy.linalg as sl
from scipy.special import loggamma

log = logging.getLogger(__name__)


def get_w_hat(eigenspectra, fluxes):
    """
    Since we will overflow memory if we actually calculate Phi, we have to
    determine w_hat in a memory-efficient manner.

    """
    m = len(eigenspectra)
    M = len(fluxes)
    out = np.empty((M * m,))
    for i in range(m):
        for j in range(M):
            out[i * M + j] = eigenspectra[i].T @ fluxes[j]

    phi_squared = get_phi_squared(eigenspectra, M)
    fac = sl.cho_factor(phi_squared, check_finite=False, overwrite_a=True)
    return sl.cho_solve(fac, out, check_finite=False)


def get_phi_squared(eigenspectra, M):
    """
    Compute Phi.T.dot(Phi) in a memory efficient manner.

    eigenspectra is a list of 1D numpy arrays.
    """
    m = len(eigenspectra)
    out = np.zeros((m * M, m * M))

    # Compute all of the dot products pairwise, beforehand
    dots = np.empty((m, m))
    for i in range(m):
        for j in range(m):
            dots[i, j] = eigenspectra[i].T @ eigenspectra[j]

    for i in range(M * m):
        for jj in range(m):
            ii = i // M
            j = jj * M + (i % M)
            out[i, j] = dots[ii, jj]
    return out


def get_altered_prior_factors(eigenspectra, fluxes):
    """
    Compute the altered priors for the :math:`\\lambda_\\xi` term as in eqns. A24 and
    A25 of Czekala et al. 2015.

    Parameters
    ----------
    eigenspectra : numpy.ndarray
        The PCA eigenspectra
    fluxes : numpy.ndarray
        The vertically stacked input spectra

    Returns
    -------
    """
    w_hat = get_w_hat(eigenspectra, fluxes)
    M, npix = fluxes.shape
    m = len(eigenspectra)

    Phi_w_hat = np.empty((M * npix, 1))
    for i in range(M):
        loss_per_M = np.zeros(npix)
        for j in range(m):
            loss_per_M += eigenspectra[j] * w_hat[i + j * M]
        indices = slice(i * npix, (i + 1) * npix)
        Phi_w_hat[indices] = loss_per_M.reshape(npix, -1)

    F = fluxes.ravel()
    a_prime = 0.5 * M * (npix - m)
    b_prime = 0.5 * (F.T @ F - F.T @ Phi_w_hat)

    return a_prime, b_prime[0]


class Gamma:
    def __init__(self, alpha, beta=1):
        self.alpha = alpha
        self.beta = beta

    def logpdf(self, x):
        lp = (
            self.alpha * np.log(self.beta)
            - loggamma(self.alpha)
            + (self.alpha - 1) * np.log(x)
            - self.beta * x
        )
        return lp

    def pdf(self, x):
        return np.exp(self.logpdf(x))
