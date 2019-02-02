from .models import SpectrumModel
from scipy.linalg import cho_factor, cho_solve
import numpy as np
import logging
from collections import deque

class SpectrumLikelihood:

    def __init__(self, spectrum, deque_length=100):
        if not isinstance(spectrum, SpectrumModel):
            raise ValueError('Must provide a valid SpectrumModel')
        self.spectrum = spectrum
        self.log = logging.getLogger(self.__class__.__name__)
        self.residuals_deque = deque(maxlen=100)

    @property
    def residuals(self):
        count = 0
        n = 0
        for r in self.residuals_deque:
            count += r
            n += 1
        return count / n

    def log_probability(self, **params):
        fls, cov = self.spectrum(**params)
        cov += np.diag(self.sigs)
        factor, flag = cho_factor(cov)

        R = self.flux - fls
        self.residuals_deque.append(R)

        logdet = 2 * np.log(np.trace(factor))
        lnprob = logdet - 0.5 * R @ cho_solve((factor, flag), R)

        self.log.debug("Evaluating lnprob={}".format(lnprob))
        return lnprob
