from collections import deque
from typing import Union, Sequence, Optional
import logging

from flatdict import FlatterDict
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import toml

from Starfish import Spectrum
from Starfish.emulator import Emulator
from Starfish.transforms import (
    rotational_broaden,
    resample,
    doppler_shift,
    extinct,
    rescale,
    _get_renorm_factor,
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
        The maximum number of residuals to retain in a deque of residuals. Default is
        100
    name : str, optional
        A name for the model. Default is 'SpectrumModel'

    Keyword Arguments
    -----------------
    params : dict
        Any remaining keyword arguments will be interpreted as parameters.


    Here is a table describing the avialable parameters and their related functions

    =========== ===============================================
     Parameter                 Function
    =========== ===============================================
    vsini        :func:`~Starfish.transforms.rotational_broaden`
    vz           :func:`~Starfish.transforms.doppler_shift`
    Av           :func:`~Starfish.transforms.extinct`
    Rv           :func:`~Starfish.transforms.extinct`
    log_scale    :func:`~Starfish.transforms.rescale`
    =========== ===============================================

    .. note::
        If :attr:`log_scale` is not specified, the model will use
        :func:`~Starfish.transforms.renorm` to automatically scale the spectrum to the
        data using the ratio of integrated fluxes.

    The ``global_cov`` keyword arguments must be a dictionary definining the
    hyperparameters for the global covariance kernel,
    :meth:`kernels.global_covariance_matrix`

    ================ =============================================================
    Global Parameter  Description
    ================ =============================================================
    log_amp          The natural logarithm of the amplitude of the Matern kernel
    log_ls           The natural logarithm of the lengthscale of the Matern kernel
    ================ =============================================================

    The ``local_cov`` keryword argument must be a list of dictionaries defining
    hyperparameters for many Gaussian kernels, , :meth:`kernels.local_covariance_matrix`

    ================ =============================================================
    Local Parameter  Description
    ================ =============================================================
    log_amp          The natural logarithm of the amplitude of the kernel
    mu               The location of the local kernel
    log_sigma        The natural logarithm of the standard deviation of the kernel
    ================ =============================================================

    Attributes
    ----------
    params : dict
        The dictionary of parameters that are used for doing the modeling.
    frozen : list
        A list of strings corresponding to frozen parameters
    residuals : deque
        A deque containing residuals from calling :meth:`SpectrumModel.log_likelihood`
    """

    _PARAMS = ["vz", "vsini", "Av", "Rv", "log_scale", "global_cov", "local_cov"]
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

        self.params = FlatterDict(params)
        self.frozen = []
        self.name = name

        # Unpack the grid parameters
        self.n_grid_params = len(grid_params)
        self.grid_params = grid_params

        self._lnprob = None
        self._glob_cov = None
        self._loc_cov = None

        self.log = logging.getLogger(self.__class__.__name__)

    @property
    def grid_params(self):
        """
        numpy.ndarray : The parameters used for the spectral emulator.

        :setter: Sets the values in the order of ``Emulator.param_names``
        """
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
        """
        tuple of str : The thawed parameter names
        """
        keys = self.get_param_dict(flat=True).keys()
        return tuple(keys)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        if ":" in key:
            cov, rest = key.split(":", 1)
            k = rest.split(":")[-1] if ":" in rest else rest
            if cov == "global_cov" and k in self._GLOBAL_PARAMS:
                self.params[key] = value
            elif cov == "local_cov" and k in self._LOCAL_PARAMS:
                self.params[key] = value
            else:
                raise ValueError(f"{key} not recognized")
        else:
            if key in [*self._PARAMS, *self.emulator.param_names]:
                self.params[key] = value
            else:
                raise ValueError(f"{key} not recognized")

    def __delitem__(self, key):
        if key not in self.params:
            raise ValueError(f"{key} not in params")
        elif key == "global_cov":
            self._glob_cov = None
            self.frozen = [
                key for key in self.frozen if not key.startswith("global_cov")
            ]
        elif key == "local_cov":
            self._loc_cov = None
            self.frozen = [
                key for key in self.frozen if not key.startswith("local_cov")
            ]

        del self.params[key]
        if key in self.frozen:
            self.frozen.remove(key)

    def __call__(self):
        """
        Performs the transformations according to the parameters available in
        ``self.params``

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
            scale = np.exp(self.params["log_scale"])
            fluxes[-2:] = rescale(fluxes[-2:], scale)

        weights, weights_cov = self.emulator(self.grid_params)

        L, flag = cho_factor(weights_cov, overwrite_a=True)

        # Decompose the bulk_fluxes (see emulator/emulator.py for the ordering)
        *eigenspectra, flux_mean, flux_std = fluxes

        # Complete the reconstruction
        X = eigenspectra * flux_std
        flux = weights @ X + flux_mean

        # Renorm to data flux if no "log_scale" provided
        if "log_scale" not in self.params:
            factor = _get_renorm_factor(self.data.wave, flux, self.data.flux)
            flux = rescale(flux, factor)
            X = rescale(X, factor)

        cov = X.T @ cho_solve((L, flag), X)

        # Trivial covariance
        np.fill_diagonal(cov, cov.diagonal() + self.data.sigma ** 2)

        # Global covariance
        if "global_cov" in self.params:
            if "global_cov" not in self.frozen or self._glob_cov is None:
                ag = np.exp(self.params["global_cov:log_amp"])
                lg = np.exp(self.params["global_cov:log_ls"])
                T = self.params["T"]
                self._glob_cov = global_covariance_matrix(self.data.wave, T, ag, lg)

        if self._glob_cov is not None:
            cov += self._glob_cov

        # Local covariance
        if "local_cov" in self.params:
            if "local_cov" not in self.frozen or self._loc_cov is None:
                for kernel in self.params.as_dict()["local_cov"]:
                    mu = kernel["mu"]
                    amplitude = np.exp(kernel["log_amp"])
                    sigma = np.exp(kernel["log_sigma"])
                    self._loc_cov = local_covariance_matrix(
                        self.data.wave, amplitude, mu, sigma
                    )

        if self._loc_cov is not None:
            cov += self._loc_cov

        return flux, cov

    def log_likelihood(self, priors: Optional[dict] = None) -> float:
        """
        Returns the log probability of a multivariate normal distribution

        Parameters
        ----------
        priors : dict, optional
            If provided, will use these priors in the MLE. Should contain keys that
            match the model's keys and values that have a `logpdf` method that takes
            one value (like ``scipy.stats`` distributions). Default is None.

        Warning
        -------
        No checks will be done on the :attr:`priors` for speed.

        Returns
        -------
        float
        """
        # Priors
        prior_lp = 0

        if priors is not None:
            for key, prior in priors.items():
                if key in self.params:
                    prior_lp += prior.logpdf(self[key])

        if not np.isfinite(prior_lp):
            return -np.inf

        # Likelihood
        flux, cov = self()
        np.fill_diagonal(cov, cov.diagonal() + 1e-10)
        factor, flag = cho_factor(cov, overwrite_a=True)
        logdet = 2 * np.sum(np.log(factor.diagonal()))
        R = flux - self.data.flux
        self.residuals.append(R)
        sqmah = R @ cho_solve((factor, flag), R)
        self._lnprob = -(logdet + sqmah) / 2

        return self._lnprob + prior_lp

    def get_param_dict(self, flat: bool = False) -> dict:
        """
        Gets the dictionary of thawed parameters.

        Parameters
        ----------
        flat : bool, optional
            If True, returns the parameters completely flat. For example,
            ``['local']['0']['mu']`` would have the key ``'local:0:mu'``.
            Default is False

        Returns
        -------
        dict

        See Also
        --------
        :meth:`set_param_dict`
        """
        params = FlatterDict()
        for key, val in self.params.items():
            if key not in self.frozen:
                params[key] = val

        return params if flat else params.as_dict()

    def set_param_dict(self, params):
        """
        Sets the parameters with a dictionary. Note that this should not be used to add
        new parameters

        Parameters
        ----------
        params : dict
            The new parameters. If a key is present in ``self.frozen`` it will not be
            changed

        See Also
        --------
        :meth:`get_param_dict`
        """
        params = FlatterDict(params)
        for key, val in params.items():
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
        Sets the parameters based on the current thawed state. The values will be
        inserted according to the order of :obj:`SpectrumModel.labels`.

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
        self.set_param_dict(param_dict)

    def freeze(self, names):
        """
        Freeze the given parameter such that :meth:`get_param_dict` and
        :meth:`get_param_vector` no longer include this parameter, however it will
        still be used when calling the model.

        Parameters
        ----------
        name : str or array-like
            The parameter to freeze. If ``'all'``, will freeze all parameters. If
            ``'global_cov'`` will freeze all global covariance parameters. If
            ``'local_cov'`` will freeze all local covariance parameters.

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
            for key in self.labels:
                if key not in self.frozen:
                    self.frozen.append(key)
            if "global_cov" in self.params:
                self.frozen.append("global_cov")
            if "local_cov" in self.params:
                self.frozen.append("local_cov")
        else:
            for _name in names:
                # Avoid kookyness of numpy.str type
                name = str(_name)
                if name == "global_cov":
                    self.frozen.append("global_cov")
                    self._glob_cov = None
                    for key in self.params.as_dict()["global_cov"].keys():
                        flat_key = f"global_cov:{key}"
                        if flat_key not in self.frozen:
                            self.frozen.append(flat_key)
                elif name == "local_cov":
                    self.frozen.append("local_cov")
                    self._loc_cov = None
                    for i, kern in enumerate(self.params.as_dict()["local_cov"]):
                        for key in kern.keys():
                            flat_key = f"local_cov:{i}:{key}"
                            if flat_key not in self.frozen:
                                self.frozen.append(flat_key)
                elif name not in self.frozen and name in self.params:
                    self.frozen.append(name)

    def thaw(self, names):
        """
        Thaws the given parameter. Opposite of freezing

        Parameters
        ----------
        name : str or array-like
            The parameter to thaw. If ``'all'``, will thaw all parameters. If
            ``'global_cov'`` will thaw all global covariance parameters. If
            ``'local_cov'`` will thaw all local covariance parameters.

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
            for _name in names:
                # Avoid kookyness of numpy.str type
                name = str(_name)
                if name == "global_cov":
                    self.frozen.remove("global_cov")
                    for key in self.params.as_dict()["global_cov"].keys():
                        flat_key = f"global_cov:{key}"
                        self.frozen.remove(flat_key)
                elif name == "local_cov":
                    self.frozen.remove("local_cov")
                    for i, kern in enumerate(self.params.as_dict()["local_cov"]):
                        for key in kern.keys():
                            flat_key = f"local_cov:{i}:{key}"
                            self.frozen.remove(flat_key)
                elif name in self.frozen:
                    self.frozen.remove(name)

    def save(self, filename, metadata=None):
        """
        Saves the model as a set of parameters into a TOML file

        Parameters
        ----------
        filename : str or path-like
            The TOML filename to save to.
        metadata : dict, optional
            If provided, will save the provided dictionary under a 'metadata' key. This
            will not be read in when loading models but provides a way of providing
            information in the actual TOML files. Default is None.
        """
        output = {"parameters": self.params.as_dict(), "frozen": self.frozen}
        meta = {}
        meta["name"] = self.name
        meta["data"] = self.data_name
        if self.emulator.name is not None:
            meta["emulator"] = self.emulator.name
        if metadata is not None:
            meta.update(metadata)
        output["metadata"] = meta

        with open(filename, "w") as handler:
            encoder = toml.TomlNumpyEncoder(output.__class__)
            toml.dump(output, handler, encoder=encoder)

        self.log.info(f"Saved current state at {filename}")

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
        self.params = FlatterDict(data["parameters"])
        self.frozen = data["frozen"]

    def train(self, priors: Optional[dict] = None, **kwargs):
        """
        Given a :class:`SpectrumModel` and a dictionary of priors, will perform
        maximum-likelihood estimation (MLE). This will use ``scipy.optimize.minimize`` to
        find the maximum a-posteriori (MAP) estimate of the current model state. Note
        that this alters the state of the model. This means that you can run this
        method multiple times until the optimization succeeds. By default, we use the
        "Nelder-Mead" method in `minimize` to avoid approximating any derivatives.

        Parameters
        ----------
        priors : dict, optional
            Priors to pass to :meth:`log_likelihood`
        **kwargs : dict, optional
            These keyword arguments will be passed to `scipy.optimize.minimize`

        Returns
        -------
        soln : `scipy.optimize.minimize_result`
            The output of the minimization.

        Raises
        ------
        ValueError
            If the priors are poorly specified
        RuntimeError
            If any priors evaluate to non-finite values

        See Also
        --------
        :meth:`log_likelihood`
        """
        if priors is None:
            priors = {}

        # Check priors for validity
        for key, val in priors.items():
            # Key exists
            if key not in self.params:
                raise ValueError(f"Invalid priors. {key} not a vlid key.")
            # has logpdf method
            if not callable(getattr(val, "logpdf", None)):
                raise ValueError(
                    f"Invalid priors. {key} does not have a `logpdf` method"
                )
            # Evaluates to a finite number in current state
            log_prob = val.logpdf(self[key])
            if not np.isfinite(log_prob):
                raise RuntimeError(f"{key}'s logpdf evaluated to {log_prob}")

        def nll(P):
            self.set_param_vector(P)
            return -self.log_likelihood(priors)

        p0 = self.get_param_vector()
        params = {"method": "Nelder-Mead"}
        params.update(kwargs)
        soln = minimize(nll, p0, **params)
        if soln.success:
            self.set_param_vector(soln.x)

        return soln

    def plot(self, axes=None, plot_kwargs=None, resid_kwargs=None):
        """
        Plot the model.

        This will create two subplots, one which shows the current model against the
        data, and another which shows the current residuals with 3:math:`\\sigma`
        contours from the diagonal of the covariance matrix. Note this requires
        matplotlib to be installed, which is not installed by default with Starfish.

        Parameters
        ----------
        axes : iterable of matplotlib.Axes, optional
            If provided, will use the first two axes to plot, otherwise will create new
            axes, by default None
        plot_kwargs : dict, optional
            If provided, will use these kwargs for the comparison plot, by default None
        resid_kwargs : dict, optional
            If provided, will use these kwargs for the residuals plot, by default None

        Returns
        -------
        list of matplotlib.Axes
            The returned axes, for the user to edit as they please
        """
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        if axes is None:
            # Set up a 4x4 grid with the main plot taking the whole left column
            figsize = rcParams["figure.figsize"]
            plt.figure(figsize=(figsize[0] * 1.75, figsize[1] * 1.1))
            grid = plt.GridSpec(2, 2, width_ratios=(1.25, 1))
            axes = [
                plt.subplot(grid[:, 0]),
                plt.subplot(grid[0, 1]),
                plt.subplot(grid[1, 1]),
            ]
            axes[1].tick_params(labelbottom=False)
        if plot_kwargs is None:
            plot_kwargs = {}
        if resid_kwargs is None:
            resid_kwargs = {}

        model_flux, model_cov = self()

        # Comparison plot
        plot_params = {"lw": 0.7}
        plot_params.update(plot_kwargs)
        ax = axes[0]
        ax.plot(self.data.wave, self.data.flux, label="Data", **plot_params)
        ax.plot(self.data.wave, model_flux, label="Model", **plot_params)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda$ [$\AA$]")
        ax.set_ylabel(r"$f_\lambda$ [$erg/cm^2/s/cm$]")
        ax.legend()

        # Residuals plot
        R = self.data.flux - model_flux
        std = np.sqrt(model_cov.diagonal())
        resid_params = {"lw": 0.3}
        resid_params.update(resid_kwargs)
        ax = axes[1]
        ax.plot(self.data.wave, R, c="k", label="Data - Model", **resid_params)
        ax.fill_between(
            self.data.wave, -std, std, color="C2", alpha=0.6, label=r"$\sigma$"
        )
        ax.fill_between(
            self.data.wave, -2 * std, 2 * std, color="C2", alpha=0.4, label=r"$2\sigma$"
        )
        ax.fill_between(
            self.data.wave, -3 * std, 3 * std, color="C2", alpha=0.2, label=r"$3\sigma$"
        )
        ax.set_ylabel(r"$\Delta f_\lambda$")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.legend()

        # Relative Error plot
        R_f = R / self.data.flux
        ax = axes[2]
        ax.plot(self.data.wave, R_f, label="Data - Model", c="k", **resid_params)
        ax.set_xlabel(r"$\lambda$ [$\AA$]")
        ax.set_ylabel(r"$\Delta f_\lambda / f_\lambda$")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        plt.suptitle(self.data_name)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return axes

    def __repr__(self):
        output = f"{self.name}\n"
        output += "-" * len(self.name) + "\n"
        output += f"Data: {self.data_name}\n"
        output += f"Emulator: {self.emulator.name}\n"
        output += f"Log Likelihood: {self._lnprob}\n"
        output += "\nParameters\n"
        for key, value in self.get_param_dict().items():
            if key == "global_cov":
                output += "  global_cov:\n"
                for gkey, gval in value.items():
                    output += f"    {gkey}: {gval}\n"
            elif key == "local_cov":
                output += "  local_cov:\n"
                for i, kern in enumerate(value.values()):
                    output += f"    {i}: "
                    for lkey, lval in kern.items():
                        output += f"{lkey}: {lval}, "
                    # Remove trailing whitespace and comma
                    output = output[:-2]
                    output += "\n"
            else:
                output += f"  {key}: {value}\n"
        if len(self.frozen) > 0:
            output += "\nFrozen Parameters\n"
            for key in self.frozen:
                if key in ["global_cov", "local_cov"]:
                    continue
                output += f"  {key}: {self[key]}\n"
        return output[:-1]  # No trailing newline
