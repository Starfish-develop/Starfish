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
            return -np.inf

        R = self.spectrum.flux - fls
        self.residuals_deque.append(R)

        logdet = 2 * np.log(np.trace(factor))
        lnprob = -0.5 * (logdet + R.T @ cho_solve((factor, flag), R) + len(R) * np.log(2 * np.pi))

        self.log.debug("Evaluating lnprob={}".format(lnprob))
        return lnprob




def optimize_spectrum_likelihood(self, **opt_params):
    pass
