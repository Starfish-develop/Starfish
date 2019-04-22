import logging
import warnings
from collections import OrderedDict, deque
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from Starfish.utils import calculate_dv, create_log_lam_grid
from .transforms import rotational_broaden, resample, doppler_shift, extinct, rescale, chebyshev_correct
from .likelihoods import mvn_likelihood, normal_likelihood
from .kernels import k_global_matrix


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

    =========== ======================================= =============
     Parameter                 Function                  Description       
    =========== ======================================= =============
    vsini        :func:`transforms.rotational_broaden`
    vz           :func:`transforms.doppler_shift`
    Av           :func:`transforms.extinct`
    Rv           :func:`transforms.extinct`              Not required. If not specified, will default to 3.1
    scale        :func:`transforms.rescale`
    log_ag       :func:`kernels.k_global_matrix`
    log_lg       :func:`kernels.k_global_matrix`
    =========== ======================================= =============

    Attributes
    ----------
    params : dict
        The dictionary of parameters that are used for doing the modeling.
    grid_params : numpy.ndarray
        A direct interface to the grid parameters rather than indexing like self['grid_param:i']
    frozen : list
        A list of strings corresponding to frozen parameters
    labels : list
        A list of strings corresponding to the active (thawed) parameters


    Raises
    ------
    ValueError
        If any of the keyword argument params do not exist in ``self._PARAMS``


    An example of adding a previously uninitialized parameter is 

    .. code-block:: python

        >>> model['<new_param>'] = new_value

    """

    _PARAMS = ['vz', 'vsini', 'Av', 'Rv',
               'global:log_ag', 'global:log_lg', 'scale']

    def __init__(self, emulator, data, grid_params, max_deque_len=100, **params):
        self.emulator = emulator

        self.data = data
        dv = calculate_dv(self.data.waves)
        self.min_dv_wave = create_log_lam_grid(
            dv, self.emulator.wl.min(), self.emulator.wl.max())['wl']
        self.bulk_fluxes = resample(
            self.emulator.wl, self.emulator.bulk_fluxes, self.min_dv_wave)

        self.residuals_queue = deque(maxlen=max_deque_len)

        self.params = OrderedDict()
        self.frozen = []

        # Unpack the grid parameters
        self.n_grid_params = len(grid_params)
        for i, value in enumerate(grid_params):
            self.params['grid_param:{}'.format(i)] = value

        # Unpack the keyword arguments
        for param, value in params.items():
            if param == 'log_ag' or param == 'log_lg':
                param = 'global:' + param
            self.params[param] = value

        self.log = logging.getLogger(self.__class__.__name__)

    @property
    def grid_params(self):
        items = [vals for key, vals in self.params.items()
                 if key.startswith('grid_param')]
        return np.array(items)

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

        if 'scale' in self.params:
            fluxes = rescale(fluxes, self.params['scale'])

        fluxes = resample(wave, fluxes, self.data.waves)

        if 'Av' in self.params:
            fluxes = extinct(self.data.waves, fluxes, self.params['Av'])

        weights, weights_cov = self.emulator(self.grid_params)

        L, flag = cho_factor(weights_cov)

        # Decompose the bulk_fluxes (see emulator/emulator.py for the ordering)
        eigenspectra = fluxes[: -2]
        flux_mean, flux_std = fluxes[-2:]

        # Complete the reconstruction
        X = eigenspectra * flux_std
        cov = X.T @ cho_solve((L, flag), X)

        # Poisson covariance
        np.fill_diagonal(cov, self.data.sigmas ** 2)

        # Global covariance
        if 'global:log_ag' in self.params and 'global:log_lg' in self.params:
            ag = np.exp(self.params['global:log_ag'])
            lg = np.exp(self.params['global:log_lg'])
            cov += k_global_matrix(self.data.waves, ag, lg)

        # Local covariance
        # TODO

        return weights @ X + flux_mean, cov

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        if not key.startswith('grid_param:') and key not in self._PARAMS:
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
        if not name in self.params:
            raise ValueError('Parameter not found')
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
        if not name in self.params:
            raise ValueError('Parameter not found')
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
        """
        Sets the parameters with a dictionary

        Parameters
        ----------
        params : dict
            The new parameters. If a key is present in ``self.frozen`` it will not be changed
        """
        for key, val in params.items():
            if key in self.params and key not in self.frozen:
                self.params[key] = val

    @property
    def labels(self):
        return list(self.get_param_dict())

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
        Sets the parameters based on the current thawed state. The values will be inserted according to the order of :obj:`SpectrumModel.labels`.

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
                'params must match length of thawed parameters (get_param_vector())')
        for i, key in enumerate(self.get_param_dict()):
            self.params[key] = params[i]

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
        self.set_param_dict(data)
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

        self.residuals_queue.append(flux - self.data.fluxes)

        lnprob = mvn_likelihood(flux, self.data.fluxes, cov)

        self.log.debug("Evaluating lnprob={}".format(lnprob))

        return lnprob

    def grad_log_likelihood(self):
        raise NotImplementedError('Not Implemented yet')


class EchelleModel:
    pass
