import logging
from copy import deepcopy
import warnings
from collections import OrderedDict, deque
import json
from typing import List, Union

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from Starfish.utils import calculate_dv, create_log_lam_grid
from .transforms import rotational_broaden, resample, doppler_shift, extinct, rescale, chebyshev_correct
from .likelihoods import mvn_likelihood, normal_likelihood
from .kernels import k_global_matrix, k_local_matrix


class SpectrumModel:
    """
    A single-order spectrum model.


    Parameters
    ----------
    emulator : :class:`Starfish.emulators.Emulator`
        The emulator to use for this model.
    data : :class:`Starfish.spectrum.DataSpectrum`
        The data to use for this model
    grid_params : array-like
        The parameters that are used with the associated emulator
    max_deque_len : int, optional
        The maximum number of residuals to retain in a deque of residuals. Default is 100

    Keyword Arguments
    -----------------
    params : dict
        Any remaining keyword arguments will be interpreted as parameters. 


    Here is a table describing the avialable parameters and their related functions

    =========== =======================================
     Parameter                 Function                
    =========== =======================================
    vsini        :func:`transforms.rotational_broaden`
    vz           :func:`transforms.doppler_shift`
    Av           :func:`transforms.extinct`
    Rv           :func:`transforms.extinct`              
    log_scale    :func:`transforms.rescale`
    =========== =======================================

    The ``glob`` keyword arguments must be a dictionary definining the hyperparameters for the global covariance kernel

    ================ =============
    Global Parameter  Description
    ================ =============
    log_amp          The natural logarithm of the amplitude of the Matern kernel
    log_ls           The natural logarithm of the lengthscale of the Matern kernel
    ================ =============

    The ``local`` keryword argument must be a list of dictionaries defining hyperparameters for many Gaussian kernels

    ================ =============
    Local Parameter  Description
    ================ =============
    log_amp          The natural logarithm of the amplitude of the kernel
    mu               The location of the local kernel
    log_sigma        The natural logarithm of the standard deviation of the kernel
    ================ =============


    Attributes
    ----------
    params : dict
        The dictionary of parameters that are used for doing the modeling.
    grid_params : numpy.ndarray
        A direct interface to the grid parameters rather than indexing like self['T']
    glob : dict
        The parameters for the global covariance kernel
    local : list of dicts
        The parameters for the local covariance kernels
    frozen : list
        A list of strings corresponding to frozen parameters
    labels : list
        A list of strings corresponding to the active (thawed) parameters
    residuals : deque
        A deque containing residuals from calling :meth:`SpectrumModel.log_likelihood`
    lnprob : float
        The most recently evaluated log-likelihood. Initialized to None

    Raises
    ------
    ValueError
        If any of the keyword argument params do not exist in ``self._PARAMS``
    """

    _PARAMS = ['vz', 'vsini', 'Av', 'Rv', 'log_scale']
    _GLOBAL_PARAMS = ['log_amp', 'log_ls']
    _LOCAL_PARAMS = ['mu', 'log_amp', 'log_sigma']

    def __init__(self, emulator, data, grid_params, max_deque_len=100, **params):
        self.emulator = emulator

        self.data = data
        dv = calculate_dv(self.data.waves)
        self.min_dv_wave = create_log_lam_grid(
            dv, self.emulator.wl.min(), self.emulator.wl.max())['wl']
        self.bulk_fluxes = resample(
            self.emulator.wl, self.emulator.bulk_fluxes, self.min_dv_wave)

        self.residuals = deque(maxlen=max_deque_len)

        self.params = OrderedDict()
        self.frozen = []
        self.lnprob = None

        # Unpack the grid parameters
        self.n_grid_params = len(grid_params)
        self.grid_params = grid_params

        # Unpack the keyword arguments
        if 'glob' in params:
            params['global'] = params.pop('glob')
        self.params.update(params)

        self.log = logging.getLogger(self.__class__.__name__)

    @property
    def grid_params(self):
        items = [vals for key, vals in self.params.items()
                 if key in self.emulator.param_names]
        return np.array(items)

    @grid_params.setter
    def grid_params(self, values):
        for i, (key, value) in enumerate(zip(self.emulator.param_names, values)):
            self.params[key] = value

    @property
    def glob(self):
        return self.params['global']

    @glob.setter
    def glob(self, params):
        if not isinstance(params, dict):
            raise ValueError('Must set global parameters with a dictionary')

        for key, value in params.items():
            if key not in self._GLOBAL_PARAMS:
                raise ValueError(
                    '{} not a recognized global parameter'.format(key))
            self.params['global'][key] = value

    @property
    def local(self):
        return self.params['local']

    @local.setter
    def local(self, params: List[dict]):
        self.params['local'] = []
        for i, kernel in enumerate(params):
            if not all([k in self._LOCAL_PARAMS for k in kernel.keys()]):
                raise ValueError('Unrecognized key in kernel {}'.format(i))
            self.params['local'].append(kernel)

    @property
    def labels(self):
        return list(self.get_param_dict(flat=True).keys())

    def __call__(self):
        """
        Performs the transformations according to the parameters available in ``self.params``

        Returns
        -------
        flux, cov : tuple
            The transformed flux and covariance matrix from the model
        """
        wave = self.min_dv_wave
        fluxes = self.bulk_fluxes

        if 'vsini' in self.params:
            fluxes = rotational_broaden(wave, fluxes, self.params['vsini'])

        if 'vz' in self.params:
            wave = doppler_shift(wave, self.params['vz'])

        fluxes = resample(wave, fluxes, self.data.waves)

        if 'Av' in self.params:
            fluxes = extinct(self.data.waves, fluxes, self.params['Av'])

        # Only rescale flux_mean and flux_std
        if 'log_scale' in self.params:
            fluxes[-2:] = rescale(fluxes[-2:], self.params['log_scale'])

        weights, weights_cov = self.emulator(self.grid_params)

        L, flag = cho_factor(weights_cov)

        # Decompose the bulk_fluxes (see emulator/emulator.py for the ordering)
        *eigenspectra, flux_mean, flux_std = fluxes

        # Complete the reconstruction
        X = eigenspectra * flux_std
        flux = weights @ X + flux_mean

        cov = X.T @ cho_solve((L, flag), X)

        # Trivial covariance
        np.fill_diagonal(cov, cov.diagonal() + self.data.sigmas ** 2)

        # Global covariance
        if 'global' in self.params:
            ag = np.exp(self.glob['log_amp'])
            lg = np.exp(self.glob['log_ls'])
            cov += k_global_matrix(self.data.waves, ag, lg)

        # Local covariance
        if 'local' in self.params:
            for kernel in self.local:
                mu = kernel['mu']
                amplitude = np.exp(kernel['log_amp'])
                sigma = np.exp(kernel['log_sigma'])
                cov += k_local_matrix(self.data.waves, amplitude, mu, sigma)

        return flux, cov

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        if key.startswith('global'):
            global_key = key.split(':')[-1]
            if global_key not in self._GLOBAL_PARAMS:
                raise ValueError(
                    '{} is not a valid global parameter.'.format(global_key))
            self.params['global'][global_key] = value
        elif key.startswith('local'):
            idx, local_key = key.split(':')[-2:]
            if local_key not in self._LOCAL_PARAMS:
                raise ValueError(
                    '{} is not a valid local parameter.'.format(local_key))
            self.params['local'][idx][local_key] = value
        else:
            if key not in self._PARAMS:
                raise ValueError('{} is not a valid parameter.'.format(key))
            self.params[key] = value

    def freeze(self, name):
        """
        Freeze the given parameter such that :meth:`get_param_dict` and :meth:`get_param_vector` no longer include this parameter, however it will still be used when calling the model.

        Parameters
        ----------
        name : str
            The parameter to freeze

        Raises
        ------
        ValueError
            If the given parameter does not exist

        See Also
        --------
        :meth:`thaw`
        """
        if name not in self.frozen:
            self.frozen.append(name)

    def thaw(self, name):
        """
        Thaws the given parameter. Opposite of freezing

        Parameters
        ----------
        name : str
            The parameter to thaw

        Raises
        ------
        ValueError
            If the given parameter does not exist.

        See Also
        --------
        :meth:`freeze`
        """
        if name in self.frozen:
            self.frozen.remove(name)

    def get_param_dict(self, flat=False):
        """
        Gets the dictionary of thawed parameters.

        flat : bool, optional
            If True, returns the parameters completely flat. For example, ['local'][0]['mu'] would have the key 'local:0:mu'. Default is False

        Returns
        -------
        dict

        See Also
        --------
        :meth:`set_param_dict`
        """
        params = {}
        for par in self.params:

            # Handle global nest
            if par == 'global':
                if not flat:
                    params['global'] = {}
                for key, val in self.params['global'].items():
                    flat_key = 'global:{}'.format(key)
                    if flat_key not in self.frozen:
                        if flat:
                            params[flat_key] = val
                        else:
                            params['global'][key] = val

            # Handle local nest
            elif par == 'local':
                # Set up list if we need to
                if not flat:
                    params['local'] = []
                for i, kernel in enumerate(self.params['local']):
                    kernel_copy = deepcopy(kernel)
                    for key, val in kernel.items():
                        flat_key = 'local:{}:{}'.format(i, key)
                        if flat_key in self.frozen:
                            del kernel_copy[key]
                        if flat and flat_key not in self.frozen:
                            params[flat_key] = val
                    if not flat:
                        params['local'].append(kernel_copy)

            # Handle base nest
            elif par not in self.frozen:
                params[par] = self.params[par]

        return params

    def set_param_dict(self, params, flat):
        """
        Sets the parameters with a dictionary. Note that this should not be used to add new parametersl

        Parameters
        ----------
        params : dict
            The new parameters. If a key is present in ``self.frozen`` it will not be changed
        flat : bool
            Whether or not the incoming dictionary is flattened

        See Also
        --------
        :meth:`get_param_dict`
        """
        for key, val in params.items():
            # Handle flat case
            if flat:
                if key not in self.frozen:
                    if key.startswith('global'):
                        global_key = key.split(':')[-1]
                        self.params['global'][global_key] = val
                    elif key.startswith('local'):
                        idx, local_key = key.split(':')[-2:]
                        self.params['local'][int(idx)][local_key] = val
                    else:
                        self.params[key] = val
            # Handle nested case
            else:
                if key == 'global':
                    for global_key, global_val in val.items():
                        flat_key = 'global:{}'.format(global_key)
                        if flat_key not in self.frozen:
                            self.params['global'][global_key] = global_val
                elif key == 'local':
                    for idx, kernel in enumerate(val):
                        for local_key, local_val in kernel.items():
                            flat_key = 'local:{}:{}'.format(idx, local_key)
                            if flat_key not in self.frozen:
                                self.params['local'][idx][local_key] = local_val
                else:
                    if key not in self.frozen:
                        self.params[key] = val

    def get_param_vector(self):
        """
        Get a numpy array of the thawed parameters

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        :meth:`set_param_vector`
        """
        return np.array(list(self.get_param_dict(flat=True).values()))

    def set_param_vector(self, params):
        """
        Sets the parameters based on the current thawed state. The values will be inserted according to the order of :obj:`SpectrumModel.labels`.

        Parameters
        ----------
        params : array_like
            The parameters to set in the model

        Raises
        ------
        ValueError
            If the `params` do not match the length of the current thawed parameters.

        See Also
        --------
        :meth:`get_param_vector`

        """
        thawed_parameters = self.labels
        if len(params) != len(thawed_parameters):
            raise ValueError(
                'params must match length of thawed parameters (get_param_vector())')
        param_dict = dict(zip(thawed_parameters, params))
        self.set_param_dict(param_dict, flat=True)

    def save(self, filename):
        """
        Saves the model as a set of parameters into a JSON file

        Parameters
        ----------
        filename : str or path-like
            The JSON filename to save to.
        """
        output = {
            **self.params,
            'frozen': self.frozen
        }
        with open(filename, 'w') as handler:
            json.dump(output, handler)
        self.log.info('Saved current state at {}'.format(filename))

    def load(self, filename):
        """
        Load a saved model state from a JSON file

        Parameters
        ----------
        filename : str or path-like
            The saved state to load
        """
        with open(filename, 'r') as handler:
            data = json.load(handler)

        frozen = data.pop('frozen')
        self.params = data
        self.frozen = frozen

    def log_likelihood(self):
        """
        Returns a multivariate normal log-likelihood

        Returns
        -------
        float
            The current log-likelihood
        """
        try:
            flux, cov = self()
        except ValueError:
            return -np.inf

        lnprob, R = mvn_likelihood(flux, self.data.fluxes, cov)
        self.residuals.append(R)
        self.log.debug("Evaluating lnprob={}".format(lnprob))
        self.lnprob = lnprob

        return lnprob

    def grad_log_likelihood(self):
        raise NotImplementedError('Not Implemented yet')

    def find_residual_peaks(self, num_residuals=100, threshold=4.0, buffer=2):
        """
        Find the peaks of the most recent residual and return their properties to aid in setting up local kernels

        Parameters
        ----------
        num_residuals : int, optional
            The number of residuals to average together for determining peaks. By default 100.
        threshold : float, optional
            The sigma clipping threshold, by default 4.0
        buffer : float, optional
            The minimum distance between peaks, in Angstrom, by default 2.0

        Returns
        -------
        means : list
            The means of the found peaks, with the same units as self.data.waves
        """
        residual = np.mean(list(self.residuals)[-num_residuals:], axis=0)
        mask = np.abs(residual - residual.mean()) > threshold * residual.std()
        peak_waves = self.data.waves[1:-1][mask[1:-1]]
        mus = []
        covered = np.zeros_like(peak_waves, dtype=bool)
        # Sort from largest residual to smallest
        abs_resid = np.abs(residual[1:-1][mask[1:-1]])
        for wl, resid in sorted(zip(peak_waves, abs_resid), key=lambda t: t[1], reverse=True):
            if wl in peak_waves[covered]:
                continue
            mus.append(wl)
            ind = (peak_waves >= (wl - buffer)) & (peak_waves <= (wl + buffer))
            covered |= ind

        return mus

    def __repr__(self):
        output = 'SpectrumModel\n'
        output += '-' * 13 + '\n'
        output += 'Data: {}\n'.format(self.data.name)
        output += 'Parameters:\n'
        for key, value in self.params.items():
            output += '\t{}: {}\n'.format(key, value)
        lnprob = self.lnprob if self.lnprob is not None else self.log_likelihood()
        output += 'Log Likelihood: {:.2f}'.format(lnprob)
        return output


class EchelleModel:
    pass
