import logging

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from Starfish.utils import calculate_dv, create_log_lam_grid
from .transforms import rotational_broaden, resample, doppler_shift, extinct, rescale, chebyshev_correct


class SpectrumModel:

    def __init__(self, emulator, data):
        self.emulator = emulator
        self.wave = data.wls[0][data.masks[0]]
        self.flux = data.fls[0][data.masks[0]]
        self.sigs = data.sigmas[0][data.masks[0]]
        dv = calculate_dv(self.wave)
        min_dv_wl = create_log_lam_grid(dv, self.emulator.wl.min(), self.emulator.wl.max())['wl']
        self.eigenspectra = resample(self.emulator.wl, self.emulator.components, min_dv_wl)

        self.log = logging.getLogger(self.__class__.__name__)

    def __call__(self, params, vsini=None, vz=None, Av=None, w=None, c=None):
        wave = self.emulator.wl
        fls = self.eigenspectra

        if vsini is not None:
            fls = rotational_broaden(wave, fls, vsini)

        if vz is not None:
            wave = doppler_shift(wave, vz)

        if Av is not None:
            fls = extinct(wave, fls, Av)

        if w is not None:
            fls = rescale(fls, w)

        if c is not None:
            fls = chebyshev_correct(fls, c)

        fls = resample(wave, fls, self.wave)

        mus, C = self.emulator(params)
        cho = cho_factor(C)
        cov = fls.T @ cho_solve(cho, fls)
        return mus @ fls, cov


class EchelleModel:
    pass
