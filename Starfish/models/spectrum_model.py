from collections import deque
import copy
import multiprocessing as mp
from typing import List, Union, Sequence, Callable, Optional
import logging

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import toml

from Starfish import Spectrum
from Starfish.emulator import Emulator
from Starfish.transforms import (
    rotational_broaden,
    resample,
    doppler_shift,
    extinct,
    rescale,
    chebyshev_correct,
)
from Starfish.utils import calculate_dv, create_log_lam_grid
from .kernels import global_covariance_matrix, local_covariance_matrix


class SpectrumModel:
    """
    A single-order spectrum model.
    Parameters
    ----------
    emulator : :class:`Starfish.emulators.Emulator`
        The emulator to use for this model.
    data : :class:`Starfish.spectrum.Spectrum`
        The data to use for this model
    grid_params : array-like
        The parameters that are used with the associated emulator
    max_deque_len : int, optional
        The maximum number of residuals to retain in a deque of residuals. Default is 100
    name : str, optional
        A name for the model. Default is 'SpectrumModel'

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
    The ``global_cov`` keyword arguments must be a dictionary definining the hyperparameters for the global covariance kernel
    ================ =============
    Global Parameter  Description
    ================ =============
    log_amp          The natural logarithm of the amplitude of the Matern kernel
    log_ls           The natural logarithm of the lengthscale of the Matern kernel
    ================ =============
    The ``local_cov`` keryword argument must be a list of dictionaries defining hyperparameters for many Gaussian kernels
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
    frozen : list
        A list of strings corresponding to frozen parameters
    labels : list
        A list of strings corresponding to the active (thawed) parameters
    residuals : deque
        A deque containing residuals from calling :meth:`SpectrumModel.log_likelihood`
    """

    _PARAMS = ["vz", "vsini", "Av", "Rv", "log_scale", "log_sigma_amp"]
    _GLOBAL_PARAMS = ["log_amp", "log_ls"]
    _LOCAL_PARAMS = ["mu", "log_amp", "log_sigma"]

    def __init__(
        self,
        emulator: Union[str, Emulator],
        data: Union[str, Spectrum],
        grid_params: Sequence[float],
        max_deque_len: int = 100,
        name: str = "SpectrumModel",
        **params,
    ):
        if isinstance(emulator, str):
            emulator = Emulator.load(emulator)
        if isinstance(data, str):
            data = Spectrum.load(data)

        if len(data) > 1:
            raise ValueError(
                "Multiple orders detected in data, please use EchelleModel"
            )

        self.emulator: Emulator = emulator
        self.data_name = data.name
        self.data = data[0]

        dv = calculate_dv(self.data.wave)
        self.min_dv_wave = create_log_lam_grid(
            dv, self.emulator.wl.min(), self.emulator.wl.max()
        )["wl"]
        self.bulk_fluxes = resample(
            self.emulator.wl, self.emulator.bulk_fluxes, self.min_dv_wave
        )

        self.residuals = deque(maxlen=max_deque_len)

        self.params = params
        self._delimiter = ":"
        self.frozen = []
        self.name = name

        # Unpack the grid parameters
        self.n_grid_params = len(grid_params)
        self.grid_params = grid_params

        self._lnprob = None

        self.log = logging.getLogger(self.__class__.__name__)

    @property
    def grid_params(self):
        values = []
        for key in self.emulator.param_names:
            values.append(self.params[key])
        return np.array(values)

    @grid_params.setter
    def grid_params(self, values):
        for key, value in zip(self.emulator.param_names, values):
            if key not in self.frozen:
                self.params[key] = value

    @property
    def labels(self):
        keys = self.get_param_dict(flat=True).keys()
        return tuple(keys)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        if key.startswith("global_cov"):
            global_key = key.split(self._delimiter)[-1]
            if global_key not in self._GLOBAL_PARAMS:
                raise ValueError(
                    "{} is not a valid global parameter.".format(global_key)
                )
            self.params["global_cov"][global_key] = value
        elif key.startswith("local_cov"):
            idx, local_key = key.split(self._delimiter)[-2:]
            if local_key not in self._LOCAL_PARAMS:
                raise ValueError("{} is not a valid local parameter.".format(local_key))
            self.params["local_cov"][int(idx)][local_key] = value
        else:
            if key not in [*self._PARAMS, *self.emulator.param_names]:
                raise ValueError("{} is not a valid parameter.".format(key))
            self.params[key] = value

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

        # Only rescale flux_mean and flux_std
        if "log_scale" in self.params:
            fluxes[-2:] = rescale(fluxes[-2:], self.params["log_scale"])

        weights, weights_cov = self.emulator(self.grid_params)

        L, flag = cho_factor(weights_cov, overwrite_a=True)

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

    def log_likelihood(self) -> float:
        """
        Returns the log probability of a multivariate normal distribution
        
        Returns
        -------
        float
        """
        try:
            flux, cov = self()
        except np.linalg.LinAlgError:
            pass
        np.fill_diagonal(cov, cov.diagonal() + 1e-10)
        factor, flag = cho_factor(cov, overwrite_a=True)
        logdet = 2 * np.sum(np.log(factor.diagonal()))
        R = flux - self.data.flux
        self.residuals.append(R)
        sqmah = R @ cho_solve((factor, flag), R)
        self._lnprob = -(logdet + sqmah) / 2
        return self._lnprob

    def get_param_dict(self, flat: bool = False) -> dict:
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
            if par == "global_cov":
                if not flat:
                    params["global_cov"] = {}
                for key, val in self.params["global_cov"].items():
                    flat_key = f"global_cov{self._delimiter}{key}"
                    if flat_key not in self.frozen:
                        if flat:
                            params[flat_key] = val
                        else:
                            params["global_cov"][key] = val

            # Handle local nest
            elif par == "local_cov":
                # Set up list if we need to
                if not flat:
                    params["local_cov"] = []
                for i, kernel in enumerate(self.params["local_cov"]):
                    kernel_copy = copy.deepcopy(kernel)
                    for key, val in kernel.items():
                        flat_key = (
                            f"local_cov{self._delimiter}{i}{self._delimiter}{key}"
                        )
                        if flat_key in self.frozen:
                            del kernel_copy[key]
                        if flat and flat_key not in self.frozen:
                            params[flat_key] = val
                    if not flat:
                        params["local_cov"].append(kernel_copy)

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
                    if key.startswith("global_cov"):
                        global_key = key.split(self._delimiter)[-1]
                        self.params["global_cov"][global_key] = val
                    elif key.startswith("local_cov"):
                        idx, local_key = key.split(self._delimiter)[-2:]
                        self.params["local_cov"][int(idx)][local_key] = val
                    else:
                        self.params[key] = val
            # Handle nested case
            else:
                if key == "global_cov":
                    for global_key, global_val in val.items():
                        flat_key = f"global_cov{self._delimiter}{global_key}"
                        if flat_key not in self.frozen:
                            self.params["global_cov"][global_key] = global_val
                elif key == "local_cov":
                    for idx, kernel in enumerate(val):
                        for local_key, local_val in kernel.items():
                            flat_key = f"local_cov{self._delimiter}{idx}{self._delimiter}{local_key}"
                            if flat_key not in self.frozen:
                                self.params["local_cov"][int(idx)][
                                    local_key
                                ] = local_val
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
        if len(params) != len(self.labels):
            raise ValueError("Param Vector does not match length of thawed parameters")
        param_dict = dict(zip(self.labels, params))
        self.set_param_dict(param_dict, flat=True)

    def freeze(self, names):
        """
        Freeze the given parameter such that :meth:`get_param_dict` and :meth:`get_param_vector` no longer include this parameter, however it will still be used when calling the model.
        Parameters
        ----------
        name : str or array-like
            The parameter to freeze. If 'all', will freeze all parameters.
        Raises
        ------
        ValueError
            If the given parameter does not exist
        See Also
        --------
        :meth:`thaw`
        """
        names = np.atleast_1d(names)
        if names[0] == "all":
            self.frozen.append(self.labels)
        else:
            for name in names:
                if name not in self.frozen:
                    self.frozen.append(name)

    def thaw(self, names):
        """
        Thaws the given parameter. Opposite of freezing
        Parameters
        ----------
        name : str or array-like
            The parameter to thaw. If 'all', will thaw all parameters
        Raises
        ------
        ValueError
            If the given parameter does not exist.
        See Also
        --------
        :meth:`freeze`
        """
        names = np.atleast_1d(names)
        if names[0] == "all":
            self.frozen = []
        else:
            for name in names:
                if name in self.frozen:
                    self.frozen.remove(name)

    def save(self, filename, metadata=None):
        """
        Saves the model as a set of parameters into a TOML file
        Parameters
        ----------
        filename : str or path-like
            The TOML filename to save to.
        metadata : dict, optional
            If provided, will save the provided dictionary under a 'metadata' key. This will not be read in when loading models but provides a way of providing information in the actual TOML files. Default is None.
        """
        output = {"parameters": self.params, "frozen": self.frozen}
        meta = {}
        meta["name"] = self.name
        meta["data"] = self.data_name
        if self.emulator.name is not None:
            meta["emulator"] = self.emulator.name
        if metadata is not None:
            meta.update(metadata)
        output["metadata"] = meta

        with open(filename, "w") as handler:
            out_str = toml.dumps(
                output, encoder=toml.TomlNumpyEncoder(output.__class__)
            )
            handler.write(out_str)

        self.log.info("Saved current state at {}".format(filename))

    def load(self, filename):
        """
        Load a saved model state from a TOML file
        Parameters
        ----------
        filename : str or path-like
            The saved state to load
        """
        with open(filename, "r") as handler:
            data = toml.load(handler)

        frozen = data["frozen"]
        self.params = data["parameters"]
        self.frozen = frozen

    def __repr__(self):
        output = f"{self.name}\n"
        output += "-" * len(self.name) + "\n"
        output += f"Data: {self.data_name}\n"
        output += f"Emulator: {self.emulator.name}\n"
        output += "Parameters:\n"
        for key, value in self.params.items():
            output += f"{key}: {value}\n"
        if self._lnprob is None:
            self.log_likelihood()  # sets the value
        output += f"\nLog Likelihood: {self._lnprob}"
        return output
