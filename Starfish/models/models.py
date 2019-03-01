import logging
import warnings
from collections import deque

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from Starfish.utils import calculate_dv, create_log_lam_grid
from .transforms import rotational_broaden, resample, doppler_shift, extinct, rescale, chebyshev_correct


class SpectrumModel:

    def __init__(self, emulator, data, grid_params, vsini=None, vz=None, Av=None, logOmega=None,
                 cheb=[1, 0, 0, 0], jitter=1e-8, deque_length=500):
        self.emulator = emulator

        if data is not None:
            mask = data.masks[0].astype(bool)
            self.wave = data.wls[0][mask]
            self.flux = data.fls[0][mask]
            self.sigs = data.sigmas[0][mask]
            dv = calculate_dv(self.wave)
            self.min_dv_wl = create_log_lam_grid(dv, self.emulator.wl.min(), self.emulator.wl.max())['wl']
            self.bulk_fluxes = resample(self.emulator.wl, self.emulator.bulk_fluxes, self.min_dv_wl)

        self.grid_params = grid_params
        self.num_grid_params = len(grid_params)
        self.vsini = vsini
        self.vz = vz
        self.Av = Av
        self.logOmega = logOmega
        self.cheb = cheb

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

    def __call__(self):
        wave = self.min_dv_wl
        fluxes = self.bulk_fluxes

        if self.vsini is not None:
            fluxes = rotational_broaden(wave, fluxes, self.vsini)

        if self.vz is not None:
            wave = doppler_shift(wave, self.vz)

        if self.Av is not None:
            fluxes = extinct(wave, fluxes, self.Av)

        if self.logOmega is not None:
            fluxes = rescale(fluxes, self.logOmega)

        # Not my favorite solution because I don't like exploiting falsiness of None
        if all(self.cheb):
            fluxes = chebyshev_correct(wave, fluxes, self.cheb)

        fluxes = resample(wave, fluxes, self.wave)
        weights, weights_cov = self.emulator(self.grid_params)
        L, flag = cho_factor(weights_cov)

        # Decompose the bulk_fluxes (see emulator/emulator.py for the ordering)
        eigenspectra = fluxes[:-2]
        flux_mean, flux_std = fluxes[-2:]

        # Complete the reconstruction
        X = eigenspectra * flux_std
        cov = X.T @ cho_solve((L, flag), X)
        return weights @ X + flux_mean, cov

    def get_param_dict(self):
        params = {
            'grid_params': self.grid_params,
            'vsini'      : self.vsini,
            'vz'         : self.vz,
            'Av'         : self.Av,
            'logOmega'   : self.logOmega,
            'cheb'       : self.cheb
        }
        return params

    def set_param_dict(self, P):
        self.grid_params = P['grid_params']
        self.vsini = P.get('vsini', None)
        self.vz = P.get('vz', None)
        self.Av = P.get('Av', None)
        self.logOmega = P.get('logOmega', None)
        self.cheb = P.get('cheb', None)

    def get_param_vector(self):
        params = self.get_param_dict()
        vector = [*params['grid_params'], params['vsini'], params['vz'], params['Av'], params['logOmega'],
                  *params['cheb'][1:]]
        return np.array(vector)

    def set_param_vector(self, P):
        self.grid_params = P[:self.num_grid_params]
        self.vsini, self.vz, self.Av, self.logOmega = P[self.num_grid_params:self.num_grid_params + 4]
        self.cheb[1:] = P[self.num_grid_params + 4:]

    def log_likelihood(self):
        try:
            fls, cov = self()
        except ValueError:
            return -np.inf
        cov += np.diag(self.sigs) + self.jitter * np.eye(len(self.wave))
        try:
            factor, flag = cho_factor(cov)
        except np.linalg.LinAlgError:
            self.log.warning('Failed to decompose covariance.')
            covariance_debugger(cov)
            return -np.inf

        R = self.flux - fls
        self.residuals_deque.append(R)

        logdet = 2 * np.log(np.trace(factor))
        lnprob = -0.5 * (logdet + R.T @ cho_solve((factor, flag), R) + len(R) * np.log(2 * np.pi))

        self.log.debug("Evaluating lnprob={}".format(lnprob))
        return lnprob

    def grad_log_likelihood(self):
        raise NotImplementedError("If you've stumbled across this, we'd love someone to calculate this gradient!")

    def save(self):
        pass

    @classmethod
    def load(cls, filename):
        pass


class EchelleModel:
    pass

log = logging.getLogger(__name__)
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
        log.info("- There are {} negative eigenvalues out of {}.".format(n_neg, n_tot))
        mark = lambda val: '>' if val < 0 else '.'

        log.info("Covariance matrix eigenvalues:")
        [log.info("{: >6} {:{fill}>20.3e}".format(i, eigenvalues[i], fill=mark(eigenvalues[i]))) for i in range(10)]
        log.info('{: >15}'.format('...'))
        [log.info("{: >6} {:{fill}>20.3e}".format(n_tot - 10 + i, eigenvalues[-10 + i],
                                                  fill=mark(eigenvalues[-10 + i]))) for i in range(10)]

    log.info('{:-^60}'.format('-'))
