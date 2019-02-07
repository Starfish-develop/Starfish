import logging

from scipy.linalg import cho_factor, cho_solve

from Starfish.utils import calculate_dv, create_log_lam_grid
from .transforms import rotational_broaden, resample, doppler_shift, extinct, rescale, chebyshev_correct


class SpectrumModel:

    def __init__(self, emulator, data):
        self.emulator = emulator
        mask = data.masks[0].astype(bool)
        self.wave = data.wls[0][mask]
        self.flux = data.fls[0][mask]
        self.sigs = data.sigmas[0][mask]
        dv = calculate_dv(self.wave)
        self.min_dv_wl = create_log_lam_grid(dv, self.emulator.wl.min(), self.emulator.wl.max())['wl']
        self.eigenspectra = resample(self.emulator.wl, self.emulator.eigenspectra, self.min_dv_wl)

        self.log = logging.getLogger(self.__class__.__name__)

    def __call__(self, parameters):
        wave = self.min_dv_wl
        fls = self.eigenspectra

        if parameters.vsini is not None:
            fls = rotational_broaden(wave, fls, parameters.vsini)

        if parameters.vz is not None:
            wave = doppler_shift(wave, parameters.vz)

        if parameters.Av is not None:
            fls = extinct(wave, fls, parameters.Av)

        if parameters.logOmega is not None:
            fls = rescale(fls, parameters.logOmega)

        # Not my favorite solution because I don't like exploiting falsiness of None
        if all(parameters.cheb):
            fls = chebyshev_correct(wave, fls, parameters.cheb)

        fls = resample(wave, fls, self.wave)
        weights, weights_cov = self.emulator(parameters.grid_params)
        cho = cho_factor(weights_cov)
        cov = fls.T @ cho_solve(cho, fls)
        return weights @ fls, cov


class EchelleModel:
    pass
