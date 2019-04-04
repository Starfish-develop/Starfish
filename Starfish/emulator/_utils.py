import logging

import numpy as np

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

    phi_squared = get_phi_squared(eigenspectra, M)

    return np.linalg.solve(phi_squared, out)


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

def inverse_block_diag(array, size):
    elements = int(array.shape[0] / size)
    output = np.empty((size, elements, elements))
    for i in range(size):
        indices = slice(i*elements, (i + 1)* elements)
        output[i] = array[indices, indices]
    return output