import numpy as np


def Phi(eigenspectra, M):
    """
    Warning: for any spectra of real-world dimensions, this routine will
    likely over flow memory.

    :param eigenspectra:
    :type eigenspectra: 2D array
    :param M: number of spectra in the synthetic library
    :type M: int
    Calculate the matrix Phi using the kronecker products.
    """

    return np.hstack([np.kron(np.eye(M), eigenspectrum[np.newaxis].T) for eigenspectrum in eigenspectra])


def get_w_hat(eigenspectra, fluxes, M):
    """
    Since we will overflow memory if we actually calculate Phi, we have to
    determine w_hat in a memory-efficient manner.

    """
    m = len(eigenspectra)
    out = np.empty((M * m,))
    for i in range(m):
        for j in range(M):
            out[i * M + j] = eigenspectra[i].T.dot(fluxes[j])

    PhiPhi = np.linalg.inv(skinny_kron(eigenspectra, M))

    return PhiPhi.dot(out)


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
            dots[i, j] = eigenspectra[i].T.dot(eigenspectra[j])

    for i in range(M * m):
        for jj in range(m):
            ii = i // M
            j = jj * M + (i % M)
            out[i, j] = dots[ii, jj]
    return out