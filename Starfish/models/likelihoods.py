import logging
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve

log = logging.getLogger(__name__)


def mvn_likelihood(fluxes, y, C):
    cov = C.copy()
    np.fill_diagonal(cov, cov.diagonal() + 1e-8)
    try:
        factor, flag = cho_factor(cov)
    except np.linalg.LinAlgError:
        log.warning(
            'Failed to decompose covariance. Entering covariance debugger')
        covariance_debugger(cov)
        sys.exit()

    R = y - fluxes

    logdet = 2 * np.log(factor.diagonal().sum())
    central = R.T @ cho_solve((factor, flag), R)
    lnprob = -0.5 * (logdet + central)
    return lnprob, R


def normal_likelihood(fluxes, y, var):
    R = y - fluxes
    l2 = R ** 2 / var
    lnprob = - 0.5 * np.sum(np.log(var) + l2)

    return lnprob, R


def covariance_debugger(cov):
    """
    Special debugging information for the covariance matrix decomposition.
    """
    log.info('{:-^60}'.format('Covariance Debugger'))
    log.info("See https://github.com/iancze/Starfish/issues/26")
    log.info("Covariance matrix at a glance:")
    if (cov.diagonal().min() < 0.0):
        log.warning("- Negative entries on the diagonal:")
        log.info("\t- Check sigAmp: should be positive")
        log.info("\t- Check uncertainty estimates: should all be positive")
    elif np.any(np.isnan(cov.diagonal())):
        log.warning("- Covariance matrix has a NaN value on the diagonal")
    else:
        if not np.allclose(cov, cov.T):
            log.warning("- The covariance matrix is highly asymmetric")

        # Still might have an asymmetric matrix below `allclose` threshold
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        n_neg = (eigenvalues < 0).sum()
        n_tot = len(eigenvalues)
        log.info(
            "- There are {} negative eigenvalues out of {}.".format(n_neg, n_tot))

        def mark(val): return '>' if val < 0 else '.'

        log.info("Covariance matrix eigenvalues:")
        [log.info("{: >6} {:{fill}>20.3e}".format(i, eigenvalues[i],
                                                  fill=mark(eigenvalues[i]))) for i in range(10)]
        log.info('{: >15}'.format('...'))
        [log.info("{: >6} {:{fill}>20.3e}".format(n_tot - 10 + i, eigenvalues[-10 + i],
                                                  fill=mark(eigenvalues[-10 + i]))) for i in range(10)]

    log.info('{:-^60}'.format('-'))
