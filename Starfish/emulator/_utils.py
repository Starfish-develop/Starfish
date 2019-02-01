import logging

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ._covariance import Sigma

log = logging.getLogger(__name__)


def get_w_hat(eigenspectra, fluxes, M):
    """
    Since we will overflow memory if we actually calculate Phi, we have to
    determine w_hat in a memory-efficient manner.

    """
    m = len(eigenspectra)
    out = np.empty((M * m,))
    for i in range(m):
        for j in range(M):
            out[i * M + j] = eigenspectra[i].T @ fluxes[j]

    PhiPhi = np.linalg.inv(skinny_kron(eigenspectra, M))

    return PhiPhi @ out


def skinny_kron(eigenspectra, M):
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


def _ln_posterior(p, emulator):
    """
    Calculate the lnprob using Habib's posterior formula for the emulator.
    """
    # We don't allow negative parameters.
    if np.any(p < 0.):
        return -np.inf

    lambda_xi = p[0]
    hyper_params = p[1:].reshape((-1, emulator.ncomps))
    variances = hyper_params[0]
    lengthscales = hyper_params[1:]

    Sig_w = Sigma(emulator.grid_points, variances, lengthscales)
    C = (1. / lambda_xi) * emulator.PhiPhi + Sig_w
    sign, logdet = np.linalg.slogdet(C)
    factor = cho_factor(C)
    central = emulator.w_hat.T @ cho_solve(factor, emulator.w_hat)
    return -0.5 * (logdet + central + emulator.grid_points.size * np.log(2. * np.pi))
