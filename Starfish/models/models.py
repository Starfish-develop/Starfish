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
            self.min_dv_wl = create_log_lam_grid(
                dv, self.emulator.wl.min(), self.emulator.wl.max())['wl']
            self.bulk_fluxes = resample(
                self.emulator.wl, self.emulator.bulk_fluxes, self.min_dv_wl)

        self.grid_params = grid_params
        self.num_grid_params = len(grid_params)
        self.vsini = vsini
        self.vz = vz
        self.Av = Av
        self.logOmega = logOmega
        self.cheb = cheb

        self.log = logging.getLogger(self.__class__.__name__)

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
        if self.cheb is not None and all(self.cheb):
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
            'vsini': self.vsini,
            'vz': self.vz,
            'Av': self.Av,
            'logOmega': self.logOmega,
            'cheb': self.cheb
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

    def save(self):
        pass

    @classmethod
    def load(cls, filename):
        pass


class EchelleModel:
    pass
