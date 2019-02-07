import logging
import sys
from collections import deque

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .models import SpectrumModel

log = logging.getLogger(__name__)


class SpectrumLikelihood:

    def __init__(self, spectrum, deque_length=100, jitter=1e-6):
        if not isinstance(spectrum, SpectrumModel):
            raise ValueError('Must provide a valid SpectrumModel')
        self.spectrum = spectrum
        self.log = logging.getLogger(self.__class__.__name__)
        self.residuals_deque = deque(maxlen=deque_length)
        self.jitter = jitter

    @property
    def residuals(self):
        count = 0
        n = 0
        for r in self.residuals_deque:
            count += r
            n += 1
        return count / n

    def log_probability(self, parameters):
        try:
            fls, cov = self.spectrum(parameters)
        except ValueError:
            return -np.inf
        cov += self.jitter * np.diag(self.spectrum.sigs)
        try:
            factor, flag = cho_factor(cov)
        except np.linalg.LinAlgError:
            self.log.warning('Failed to decompose covariance. Entering covariance debugger')
            covariance_debugger(cov)
            sys.exit()

        R = self.spectrum.flux - fls
        self.residuals_deque.append(R)

        logdet = 2 * np.log(np.trace(factor))
        lnprob = -0.5 * (logdet + R.T @ cho_solve((factor, flag), R) + len(R) * np.log(2 * np.pi))

        self.log.debug("Evaluating lnprob={}".format(lnprob))
        return lnprob


def covariance_debugger(cov):
    '''
    Special debugging information for the covariance matrix decomposition.
    '''
    log.setLevel(logging.INFO)
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
        log.info("- There are {} negative eigenvalues out of {}.".format(n_neg, n_tot))
        mark = lambda val: '>' if val < 0 else '.'

        log.info("Covariance matrix eigenvalues:")
        [log.info("{: >6} {:{fill}>20.3e}".format(i, eigenvalues[i], fill=mark(eigenvalues[i]))) for i in range(10)]
        log.info('{: >15}'.format('...'))
        [log.info("{: >6} {:{fill}>20.3e}".format(n_tot - 10 + i, eigenvalues[-10 + i],
                                                  fill=mark(eigenvalues[-10 + i]))) for i in range(10)]

    log.info('{:-^60}'.format('-'))


def optimize_spectrum_likelihood(self, **opt_params):
    pass
