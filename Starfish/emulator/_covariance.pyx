# encoding: utf-8
# cython: profile=True
# filename: _covariance.pyx

cimport numpy as np
cimport cython
import numpy as np
from scipy.linalg import block_diag
import math
from scipy.spatial.distance import cdist

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
    # The covariance only depends on the distance squared
    return variance * np.exp(-0.5 * cdist(X / lengthscale, Z / lengthscale, 'sqeuclidean'))



def block_kernel(np.ndarray[np.double_t, ndim=2] X, np.ndarray[np.double_t, ndim=2] Z,
                 np.ndarray[np.double_t, ndim=1] variances, np.ndarray[np.double_t, ndim=2] lengthscales):
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

    blocks = [rbf_kernel(X, Z, variances[block], lengthscales[block]) for block in range(m)]
    return block_diag(*blocks)
