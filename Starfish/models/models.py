import logging
import warnings
from collections import OrderedDict
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from Starfish.utils import calculate_dv, create_log_lam_grid
from .transforms import rotational_broaden, resample, doppler_shift, extinct, rescale, chebyshev_correct


class SpectrumModel:

    def __init__(self, emulator, data, grid_params, vsini=None, vz=None, Av=None, scale=None):
        self.emulator = emulator

        mask = data.masks[0].astype(bool)
        self.wave = data.wls[0][mask]
        self.flux = data.fls[0][mask]
        self.sigs = data.sigmas[0][mask]
        dv = calculate_dv(self.wave)
        self.min_dv_wl = create_log_lam_grid(
            dv, self.emulator.wl.min(), self.emulator.wl.max())['wl']
        self.bulk_fluxes = resample(
            self.emulator.wl, self.emulator.bulk_fluxes, self.min_dv_wl)

        self.params = OrderedDict()
        self.frozen = []
        # Unpack the grid parameters
        self.n_grid_params = len(grid_params)
        for i, value in enumerate(grid_params):
            self.params['grid_param:{}'.format(i)] = value

        if vsini is not None:
            self.params['vsini'] = vsini

        if vz is not None:
            self.params['vz'] = vz

        if Av is not None:
            self.params['Av'] = Av

        if scale is not None:
            self.params['scale'] = scale

        self.log = logging.getLogger(self.__class__.__name__)

    @property
    def grid_params(self):
        items = [vals for key, vals in self.params.items()
                 if key.startswith('grid_param')]
        return np.array(items)

    def __call__(self):
        wave = self.min_dv_wl
        fluxes = self.bulk_fluxes

        if 'vsini' in self.params:
            fluxes = rotational_broaden(wave, fluxes, self.params['vsini'])

        if 'vz' in self.params:
            wave = doppler_shift(wave, self.params['vz'])

        if 'scale' in self.params:
            fluxes = rescale(fluxes, self.params['scale'])

        # Not my favorite solution because I don't like exploiting falsiness of None
        # if 'cheb' in self.params:
        #     fluxes = chebyshev_correct(wave, fluxes, self.cheb)

        fluxes = resample(wave, fluxes, self.wave)

        if 'Av' in self.params:
            fluxes = extinct(self.wave, fluxes, self.params['Av'])

        weights, weights_cov = self.emulator(self.grid_params)
        L, flag = cho_factor(weights_cov)

        # Decompose the bulk_fluxes (see emulator/emulator.py for the ordering)
        eigenspectra = fluxes[: -2]
        flux_mean, flux_std = fluxes[-2:]

        # Complete the reconstruction
        X = eigenspectra * flux_std
        cov = X.T @ cho_solve((L, flag), X)
        return weights @ X + flux_mean, cov

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

    def freeze(self, name):
        if name not in self.frozen:
            self.frozen.append(name)

    def thaw(self, name):
        if name in self.frozen:
            self.frozen.remove(name)

    def get_param_dict(self):
        """
        Gets the dictionary of thawed parameters.

        Returns
        -------
        dict
        """
        params = {}
        for par in self.params:
            if par not in self.frozen:
                params[par] = self.params[par]
        return params

    def set_param_dict(self, params):
        for key, val in params.items():
            if key in self.params and key not in self.frozen:
                self.params[key] = val

    def get_param_vector(self):
        """
        Get a numpy array of the thawed parameters

        Returns
        -------
        numpy.ndarray
        """
        return np.array(list(self.get_param_dict().values()))

    def set_param_vector(self, params):
        """
        Sets the parameters based on the current thawed state. The values will be inserted according to the order of :function:`SpectrumModel.get_param_dict()`.

        Parameters
        ----------
        params : array_like
            The parameters to set in the model

        Raises
        ------
        ValueError
            If the `params` do not match the length of the current thawed parameters.

        """
        thawed_parameters = self.get_param_dict()
        if len(params) != len(thawed_parameters):
            raise ValueError(
                'params must match length of thawed parameters (get_param_dict())')
        for i, key in enumerate(self.get_param_dict()):
            self.params[key] = params[i]

    def save(self, filename):
        output = {
            **self.params,
            'frozen': self.frozen
        }
        with open(filename, 'w') as handler:
            json.dump(output, handler)
        self.log.info('Saved current state at {}'.format(filename))

    def load(self, filename):
        with open(filename, 'r') as handler:
            data = json.load(handler)

        frozen = data.pop('frozen')
        self.set_param_dict(data)
        self.frozen = frozen
        


class EchelleModel:
    pass
