import h5py
import numpy as np
from dataclasses import dataclass
from nptyping import NDArray
from typing import Optional


@dataclass
class Order:
    """
    A data class to hold astronomical spectra orders.

    Parameters
    ----------
    _wave : numpy.ndarray
        The full wavelength array
    _flux : numpy.ndarray
        The full flux array
    _sigma : numpy.ndarray, optional
        The full sigma array. If None, will default to all 0s. Default is None
    mask : numpy.ndarray, optional
        The full mask. If None, will default to all Trues. Default is None

    Attributes
    ----------
    name : str
    """

    _wave: NDArray[float]
    _flux: NDArray[float]
    _sigma: Optional[NDArray[float]] = None
    mask: Optional[NDArray[bool]] = None

    def __post_init__(self):
        if self._sigma is None:
            self._sigma = np.zeros_like(self._flux)
        if self.mask is None:
            self.mask = np.ones_like(self._wave, dtype=bool)

    @property
    def wave(self):
        """
        numpy.ndarray : The masked wavelength array
        """
        return self._wave[self.mask]

    @property
    def flux(self):
        """
        numpy.ndarray : The masked flux array
        """
        return self._flux[self.mask]

    @property
    def sigma(self):
        """
        numpy.ndarray : The masked flux uncertainty array
        """
        return self._sigma[self.mask]

    def __len__(self):
        return len(self._wave)


class Spectrum:
    """
    Object to store astronomical spectra.

    Parameters
    ----------
    waves : 1D or 2D array-like
        wavelength in Angtsrom
    fluxes : 1D or 2D array-like
         flux (in f_lam)
    sigmas : 1D or 2D array-like, optional
        Poisson noise (in f_lam). If not specified, will be zeros. Default is None
    masks : 1D or 2D array-like, optional
        Mask to blot out bad pixels or emission regions. Must be castable to boolean. If None, will create a mask of all True. Default is None
    name : str, optional
        The name of this spectrum. Default is "Spectrum"

    Note
    ----
    If the waves, fluxes, and sigmas are provided as 1D arrays (say for a single order), they will be converted to 2D arrays with length 1 in the 0-axis.

    Warning
    -------
    For now, the Spectrum waves, fluxes, sigmas, and masks must be a rectangular grid. No ragged Echelle orders allowed.

    Attributes
    ----------
    name : str
        The name of the spectrum
    """

    def __init__(self, waves, fluxes, sigmas=None, masks=None, name="Spectrum"):
        waves = np.atleast_2d(waves)
        fluxes = np.atleast_2d(fluxes)

        if sigmas is not None:
            sigmas = np.atleast_2d(sigmas)
        else:
            sigmas = np.ones_like(fluxes)

        if masks is not None:
            masks = np.atleast_2d(masks).astype(bool)
        else:
            masks = np.ones_like(waves, dtype=bool)
        assert fluxes.shape == waves.shape, "flux array incompatible shape."
        assert sigmas.shape == waves.shape, "sigma array incompatible shape."
        assert masks.shape == waves.shape, "mask array incompatible shape."
        self.orders = []
        for i in range(len(waves)):
            self.orders.append(Order(waves[i], fluxes[i], sigmas[i], masks[i]))
        self.name = name

    def __getitem__(self, index: int):
        return self.orders[index]

    def __setitem__(self, index: int, order: Order):
        if len(order) != len(self.orders[0]):
            raise ValueError("Invalid order length; no ragged spectra allowed")
        self.orders[index] = order

    def __len__(self):
        return len(self.orders)

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < len(self.orders):
            n, self._n = self._n, self._n + 1
            return self.orders[n]
        else:
            raise StopIteration

    # Masked properties
    @property
    def waves(self) -> np.ndarray:
        """
        numpy.ndarray : The 2 dimensional masked wavelength arrays
        """
        waves = [o.wave for o in self.orders]
        return np.asarray(waves)

    @property
    def fluxes(self) -> np.ndarray:
        """
        numpy.ndarray : The 2 dimensional masked flux arrays
        """
        fluxes = [o.flux for o in self.orders]
        return np.asarray(fluxes)

    @property
    def sigmas(self) -> np.ndarray:
        """
        numpy.ndarray : The 2 dimensional masked flux uncertainty arrays
        """
        sigmas = [o.sigma for o in self.orders]
        return np.asarray(sigmas)

    # Unmasked properties
    @property
    def _waves(self) -> np.ndarray:
        _waves = [o._wave for o in self.orders]
        return np.asarray(_waves)

    @property
    def _fluxes(self) -> np.ndarray:
        _fluxes = [o._flux for o in self.orders]
        return np.asarray(_fluxes)

    @property
    def _sigmas(self) -> np.ndarray:
        _sigmas = [o._sigma for o in self.orders]
        return np.asarray(_sigmas)

    @property
    def masks(self) -> np.ndarray:
        """
        np.ndarray: The full 2-dimensional boolean masks
        """
        waves = [o.wave for o in self.orders]
        return np.asarray(waves)

    @property
    def shape(self):
        """
        numpy.ndarray: The shape of the spectrum, *(norders, npixels)*

        :setter: Tries to reshape the data into a new arrangement of orders and pixels following numpy reshaping rules.
        """
        return (len(self), len(self.orders[0]))

    @shape.setter
    def shape(self, shape):
        new = self.reshape(shape)
        self.__dict__.update(new.__dict__)

    def reshape(self, shape):
        """
        Reshape the spectrum to the new shape. Obeys the same rules that numpy reshaping does. Note this is not done in-place.

        Parameters
        ----------
        shape : tuple
            The new shape of the spectrum. Must abide by numpy reshaping rules.

        Returns
        -------
        Spectrum
            The reshaped spectrum
        """
        waves = self._waves.reshape(shape)
        fluxes = self._fluxes.reshape(shape)
        sigmas = self._sigmas.reshape(shape)
        masks = self.masks.reshape(shape)
        return self.__class__(waves, fluxes, sigmas, masks, name=self.name)

    @classmethod
    def load(cls, filename):
        """
        Load a spectrum from an hdf5 file

        Parameters
        ----------
        filename : str or path-like
            The path to the HDF5 file.

        See Also
        --------
        :meth:`save`
        """
        with h5py.File(filename, "r") as base:
            if "name" in base.attrs:
                name = base.attrs["name"]
            else:
                name = None
            waves = base["waves"][:]
            fluxes = base["fluxes"][:]
            sigmas = base["sigmas"][:]
            masks = base["masks"][:]
        return cls(waves, fluxes, sigmas, masks, name=name)

    def save(self, filename):
        """
        Takes the current DataSpectrum and writes it to an HDF5 file.

        Parameters
        ----------
        filename: str or path-like
            The filename to write to. Will not create any missing directories.

        See Also
        --------
        :meth:`load`
        """

        with h5py.File(filename, "w") as base:
            base.create_dataset("waves", data=self.waves, compression=9)
            base.create_dataset("fluxes", data=self.fluxes, compression=9)
            base.create_dataset("sigmas", data=self.sigmas, compression=9)
            base.create_dataset("masks", data=self.masks, compression=9)
            if self.name is not None:
                base.attrs["name"] = self.name

    def plot(self, ax=None, **kwargs):
        """
        Plot all the orders of the spectrum

        Parameters
        ----------
        ax : matplotlib.Axes, optional
            If provided, will plot on this axis. Otherwise, will create a new axis, by
            default None

        Returns
        -------
        matplotlib.Axes
            The axis that was plotted on
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        plot_params = {"lw": 0.5}
        plot_params.update(kwargs)
        # Plot orders
        for i, order in enumerate(self.orders):
            ax.plot(order._wave, order._flux, label=f"Order: {i}", **plot_params)

        # Now plot masks
        ylims = ax.get_ylim()
        for i, order in enumerate(self.orders):
            ax.fill_between(
                order._wave,
                *ylims,
                color="k",
                alpha=0.1,
                where=~order.mask,
                label="Mask" if i == 0 else None,
            )

        ax.set_yscale("log")
        ax.set_ylabel(r"$f_\lambda$ [$erg/cm^2/s/cm$]")
        ax.set_xlabel(r"$\lambda$ [$\AA$]")
        ax.legend()
        if self.name is not None:
            ax.set_title(self.name)
        fig.tight_layout()
        return ax

    def __repr__(self):
        return f"{self.name} ({self.waves.shape[0]} orders)"
