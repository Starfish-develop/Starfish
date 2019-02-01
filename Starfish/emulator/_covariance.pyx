# encoding: utf-8
# cython: profile=True
# filename: _covariance.pyx

import numpy as np
from scipy.linalg import block_diag
cimport numpy as np
cimport cython
import math

# Routines for emulator setup

@cython.boundscheck(False)
cdef rbf_kernel(np.ndarray[np.double_t, ndim=1] X, np.ndarray[np.double_t, ndim=1] Z, double variance,
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
    dp = (X - Z) ** 2  # The covariance only depends on the distance squared
    return variance * math.exp(-0.5 * np.sum(dp / lengthscale))

@cython.boundscheck(False)
def sigma(np.ndarray[np.double_t, ndim=2] grid_points, double variance, np.ndarray[np.double_t, ndim=1] lengthscale):
    """
    Compute the Lambda block of covariance for a single eigenspectrum weight.

    Parameters
    ----------
    grid_points : numpy.ndarray
        Parameters at which the synthetic grid provides spectra
    variance : double
        The variance for the kernel
    lengthscale : numpy.ndarray
        The lengthscale vector for the kernel

    Returns
    -------
    numpy.ndarray (len(grid_points), len(grid_points))
    """
    cdef int m = len(grid_points)
    cdef int i = 0
    cdef int j = 0
    cdef double cov = 0.0

    cdef np.ndarray[np.double_t, ndim=2] mat = np.empty((m, m), dtype=np.float64)

    for i in range(m):
        for j in range(i + 1):
            cov = rbf_kernel(grid_points[i], grid_points[j], variance, lengthscale)
            mat[i, j] = cov
            mat[j, i] = cov

    return mat

def Sigma(np.ndarray[np.double_t, ndim=2] grid_points, np.ndarray[np.double_t, ndim=1] variances,
          np.ndarray[np.double_t, ndim=2] lengthscales):
    """
    Fill in the large Sigma matrix using blocks of smaller sigma matrices
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
    sig_list = []
    m = len(variances)

    for variance, lengthscale in zip(variances, lengthscales):
        sig_list.append(sigma(grid_points, variance, lengthscale))

    return block_diag(*sig_list)

def V12(np.ndarray[np.double_t, ndim=1] params, np.ndarray[np.double_t, ndim=2] grid_points,
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
    cdef int M = len(grid_points)
    cdef int m = len(variances)

    mat = np.zeros((m * M, m), dtype=np.double)
    for block in range(m):
        for row in range(M):
            mat[block * M + row, block] = rbf_kernel(grid_points[row], params, variances[block], lengthscales[block])
    return mat

def V12m(np.ndarray[np.double_t, ndim=2] params, np.ndarray[np.double_t, ndim=2] grid_points,
         np.ndarray[np.double_t, ndim=1] variances, np.ndarray[np.double_t, ndim=2] lengthscales):
    """
    Calculate V12 for a multiple parameter values.

    Parameters
    ----------
    params
    grid_points
    variances
    lengthscales

    Returns
    -------

    """
    cdef int M = len(grid_points)
    cdef int npar = len(params)
    cdef int m = len(variances)

    mat = np.zeros((m * M, m * npar), dtype=np.float64)

    # Going down the rows in "blocks" corresponding to the eigenspectra
    for block in range(m):
        # Now go down the rows within that block
        for row in range(M):
            ii = block * M + row
            # Now go across the columns within that row
            for ip in range(npar):
                jj = block + ip * m
                mat[ii, jj] = rbf_kernel(grid_points[row], params[ip], variances[block], lengthscales[block])
    return mat

def V22(np.ndarray[np.double_t, ndim=1] params, np.ndarray[np.double_t, ndim=1] variances,
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
    cdef int i = 0
    cdef int m = len(variances)

    mat = np.zeros((m, m), dtype=np.double)
    for i in range(m):
        mat[i, i] = rbf_kernel(params, params, variances[i], lengthscales[i])
    return mat

def V22m(np.ndarray[np.double_t, ndim=2] params, np.ndarray[np.double_t, ndim=1] variances,
         np.ndarray[np.double_t, ndim=2] lengthscales):
    """
    Create V22 for a set of many parameters.

    Parameters
    ----------
    params
    variances
    lengthscales

    Returns
    -------

    """
    cdef int i = 0
    cdef int m = len(variances)
    cdef int npar = len(params)
    cdef double cov = 0.0

    mat = np.zeros((m * npar, m * npar))
    for ixp in range(npar):
        for i in range(m):
            for iyp in range(npar):
                ii = ixp * m + i
                jj = iyp * m + i
                cov = rbf_kernel(params[ixp], params[iyp], variances[i], lengthscales[i])
                mat[ii, jj] = cov
                mat[jj, ii] = cov
    return mat
# Routines for data covariance matrix generation


# TODO refactor these into the Models package
