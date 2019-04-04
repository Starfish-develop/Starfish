import scipy as sp
import numpy as np


def rbf_kernel(X, Z, variance, lengthscale):
    """
    A classic radial-basis function (Gaussian; exponential squared) covariance kernel

    .. math::
        \\kappa(X, Z | \\sigma^2, \\Lambda) = \\sigma^2 \\exp\\left[-\\frac12 (X-Z)^T \\Lambda^{-1} (X-Z) \\right]

    Parameters
    ----------
    X : np.ndarray
        The first set of points
    Z : np.ndarray
        The second set of points. Must have same second dimension as `X`
    variance : double
        The amplitude for the RBF kernel
    lengthscale : np.ndarray or double
        The lengthscale for the RBF kernel. Must have same second dimension as `X`

    """

    sq_dist = sp.spatial.distance.cdist(
        X / lengthscale, Z / lengthscale, 'sqeuclidean')
    return variance * np.exp(-0.5 * sq_dist)


def batch_kernel(X, Z, variances, lengthscales):
    """
    Batched RBF kernel

    Parameters
    ----------
    X : np.ndarray
        The first set of points
    Z : np.ndarray
        The second set of points. Must have same second dimension as `X`
    variances : np.ndarray
        The amplitude for the RBF kernel
    lengthscales : np.ndarray
        The lengthscale for the RBF kernel. Must have same second dimension as `X`

    See Also
    --------
    :function:`rbf_kernel`
    """
    blocks = [rbf_kernel(X, Z, var, ls)
              for var, ls in zip(variances, lengthscales)]
    return sp.linalg.block_diag(*blocks)
