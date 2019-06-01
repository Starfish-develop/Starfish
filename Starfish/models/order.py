from collections import deque
import copy
import logging
import toml
from typing import Sequence, Union, List

from flatdict import FlatterDict
from nptyping import Array
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from Starfish import Emulator
from Starfish.utils import calculate_dv, create_log_lam_grid
from .transforms import (
    rotational_broaden,
    resample,
    doppler_shift,
    extinct,
    rescale,
    chebyshev_correct,
)
from .likelihoods import order_likelihood
from .kernels import global_covariance_matrix, local_covariance_matrix


class OrderModel:
    """
    A model for a single order of data

    Parameters
    ----------
    emulator : :class:`Starfish.Emulator`
        The emulator to use for this model.
    data : :class:`Starfish.Spectrum`
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
    cheb         :func:`transforms.chebyshev_correct`
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
    cheb_params : numpy.ndarray
        A direct interface to the chebyshev parameters
    frozen : list
        A list of strings corresponding to frozen parameters
    labels : list
        A list of strings corresponding to the active (thawed) parameters
    residuals : deque
        A deque containing residuals from calling :meth:`SpectrumModel.log_likelihood`

    Raises
    ------
    ValueError
        If any of the keyword argument params do not exist in ``self._PARAMS``
    """

    _PARAMS = [
        "vz",
        "vsini",
        "Av",
        "Rv",
        "log_scale",
        "log_sigma_amp",
        "cheb",
        "log_amp",
        "log_ls",
        "mu",
        "log_sigma",
    ]

    def __init__(
        self,
        emulator: Union[str, Emulator],
        order: "Starfish.models.OrderModel",
        grid_params: Sequence[float],
        max_deque_len: int = 100,
        name: str = "OrderModel",
        **params,
    ):

        if isinstance(emulator, str):
            emulator = Emulator.load(emulator)

        self.emulator = emulator
        self.data = order

        self.params = FlatterDict()
        self.frozen = []
        self.name = name
        self.residuals = deque(maxlen=max_deque_len)

        # Unpack the grid parameters
        self.n_grid_params = len(grid_params)
        self.grid_params = grid_params

        self.log = logging.getLogger(self.__class__.__name__)

        dv = calculate_dv(self.data.wave)

        self.min_dv_wave = create_log_lam_grid(
            dv, self.emulator.wl.min(), self.emulator.wl.max()
        )["wl"]
        self.bulk_fluxes = resample(
            self.emulator.wl, self.emulator.bulk_fluxes, self.min_dv_wave
        )
        # Unpack the keyword arguments
        self.params.update(params)

    @property
    def grid_params(self):
        items = [
            vals
            for key, vals in self.params.items()
            if key in self.emulator.param_names
        ]
        return np.array(items)

    @grid_params.setter
    def grid_params(self, values):
        for i, (key, value) in enumerate(zip(self.emulator.param_names, values)):
            self.params[key] = value

    @property
    def independent_params(self):
        return FlatterDict(
            {
                **self.params["cheb"],
                **self.params["global_cov"],
                **self.params["local_cov"],
            }
        )

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

        if "vsini" in self.params:
            fluxes = rotational_broaden(wave, fluxes, self.params["vsini"])

        if "vz" in self.params:
            wave = doppler_shift(wave, self.params["vz"])

        fluxes = resample(wave, fluxes, self.data.wave)

        if "Av" in self.params:
            fluxes = extinct(self.data.wave, fluxes, self.params["Av"])

        if "cheb" in self.params:
            fluxes = chebyshev_correct(self.data.wave, fluxes, self.params["cheb"])

        # Only rescale flux_mean and flux_std
        if "log_scale" in self.params:
            fluxes[-2:] = rescale(fluxes[-2:], self.params["log_scale"])

        weights, weights_cov = self.emulator(self.grid_params)

        L, flag = cho_factor(weights_cov)

        # Decompose the bulk_fluxes (see emulator/emulator.py for the ordering)
        *eigenspectra, flux_mean, flux_std = fluxes

        # Complete the reconstruction
        X = eigenspectra * flux_std
        flux = weights @ X + flux_mean

        cov = X.T @ cho_solve((L, flag), X)

        # Trivial covariance
        np.fill_diagonal(cov, cov.diagonal() + self.data.sigma ** 2)

        # Global covariance
        if "global_cov" in self.params:
            ag = np.exp(self.params["global_cov"]["log_amp"])
            lg = np.exp(self.params["global_cov"]["log_ls"])
            cov += global_covariance_matrix(self.data.wave, ag, lg)

        # Local covariance
        if "local_cov" in self.params:
            for kernel in self.params["local_cov"]:
                mu = kernel["mu"]
                amplitude = np.exp(kernel["log_amp"])
                sigma = np.exp(kernel["log_sigma"])
                cov += local_covariance_matrix(self.data.wave, amplitude, mu, sigma)

        return flux, cov

    def __getitem__(self, key: str):
        return self.params[key]

    def __setitem__(self, key: str, value):
        if not key.split(":")[-1] in self._PARAMS:
            raise ValueError(f"{key} not a recognized parameter.")
        self.params[key] = value

    def log_likelihood(self):
        return order_likelihood(self)

    def get_param_dict(self, flat: bool = False) -> Union[FlatterDict, dict]:
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
        params = copy.deepcopy(self.params)
        for key in self.frozen:
            del params[key]

        return params if flat else params.as_dict()

    def set_param_dict(self, params: dict):
        """
        Sets the parameters with a dictionary. Note that this should not be used to add new parametersl

        Parameters
        ----------
        params : dict
            The new parameters. If a key is present in ``self.frozen`` it will not be changed

        See Also
        --------
        :meth:`get_param_dict`
        """
        if not isinstance(params, FlatterDict):
            params = FlatterDict(params)
        for key in self.frozen:
            del params[key]
        self.params.update(params)

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
                "params must match length of thawed parameters (get_param_vector())"
            )
        param_dict = dict(zip(thawed_parameters, params))
        self.set_param_dict(param_dict)

    def __repr__(self):
        output = f"{self.name}\n"
        output += "-" * len(self.name) + "\n"
        output += f"Data: {self.data.name}\n"
        output += "Parameters:\n"
        params = self.params.as_dict()
        for key, value in params.items():
            if key != "global_cov" and key != "local_cov":
                output += f"\t{key}: {value}\n"

        if "global_cov" in params:
            output += "Global Parameters:\n"
            for key, value in params["global_cov"].items():
                output += f"\t{key}: {value}\n"

        if "local_cov" in params:
            output += "Local Parameters:\n"
            for i, kernel in enumerate(params["local_cov"]):
                output += f"\t{i}: "
                for key, value in kernel.items():
                    output += f"{key}: {value}\n\t   "
                output = output[:-4]

        output += f"\nLog Likelihood: {self.log_likelihood()}"
        return output
