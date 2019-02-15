# encoding: utf-8
# cython: profile=True
# filename: _covariance.pyx

cimport numpy as np
cimport cython
import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist

# Routines for emulator setup

@cython.boundscheck(False)
cdef rbf_kernel(np.ndarray[np.double_t, ndim=2] X, np.ndarray[np.double_t, ndim=2] Z, double variance,
                np.ndarray[np.double_t, ndim=1] lengthscale):
    """
    A classic RBF kernel with ARD parameters
    
    .. math:: 
        \\Kappa (X, Z | \\sigma^2, \\Lambda) = \\sigma^2 \\exp \\left[ - \\frac12 (X-Z)^T\\Lambda^{-1}(X-Z) \\right] 
    
    where :math:`\\sigma^2` is `variance` and :math:`\\Lambda` is `lengthscale`
    
    Parameters
    ----------
    X : numpy.ndarray
        The first set of points for the kernel 
    Z : numpy.ndarray
        The second set of points for the kernel
    variance : double
        The variance for the kernel
    lengthscale : numpy.ndarray
        The lengthscale vector for the kernel

    Returns
    -------
    double

    """
    X_scaled = X / lengthscale
    Z_scaled = Z / lengthscale
    # The covariance only depends on the distance squared
    cdef np.ndarray[np.double_t, ndim=2] R2 = cdist(X_scaled, Z_scaled, 'sqeuclidean')
    return variance * np.exp(-0.5 * R2)

def block_sigma(np.ndarray[np.double_t, ndim=2] grid_points, np.ndarray[np.double_t, ndim=1] variances,
                np.ndarray[np.double_t, ndim=2] lengthscales):
    """
    Fill in the large block_sigma matrix using blocks of smaller sigma matrices
    Parameters
    ----------
    grid_points : numpy.ndarray
        Parameters at which the synthetic grid provides spectra
    variance : numpy.ndarray
        The variance for the kernel
    lengthscale : numpy.ndarray
        The lengthscale vector for the kernel

    Returns
    -------
    numpy.ndarray (m * len(grid_points), m * len(grid_points))

    """
    cdef int m = len(variances)

    blocks = [rbf_kernel(grid_points, grid_points, variances[block], lengthscales[block]) for block in range(m)]
    return block_diag(*blocks)

def V12(np.ndarray[np.double_t, ndim=2] grid_points, np.ndarray[np.double_t, ndim=2] params,
        np.ndarray[np.double_t, ndim=1] variances, np.ndarray[np.double_t, ndim=2] lengthscales):
    """
    Calculate V12 for a single parameter value.

    Parameters
    ----------
    params
    grid_points
    variances
    lengthscales

    Returns
    -------

    """
    cdef int m = len(variances)

    blocks = [rbf_kernel(grid_points, params, variances[block], lengthscales[block]) for block in range(m)]
    return block_diag(*blocks)

def V22(np.ndarray[np.double_t, ndim=2] params, np.ndarray[np.double_t, ndim=1] variances,
        np.ndarray[np.double_t, ndim=2] lengthscales):
    """
    Create V22.

    Parameters
    ----------
    params
    variances
    lengthscales

    Returns
    -------

    """
    cdef int m = len(variances)

    blocks = [rbf_kernel(params, params, variances[block], lengthscales[block]) for block in range(m)]
    return block_diag(*blocks)
