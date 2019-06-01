import h5py
import numpy as np
from dataclasses import dataclass
from nptyping import Array
from typing import Optional, Union


@dataclass
class Order:
    _wave: Array[float]
    _flux: Array[float]
    _sigma: Optional[Array[float]] = None
    mask: Optional[Array[bool]] = None

    def __post_init__(self):
        if self._sigma is None:
            self._sigma = np.ones_like(self._flux)
        if self.mask is None:
            self.mask = np.ones_like(self._wave, dtype=bool)

    @property
    def wave(self):
        return self._wave[self.mask]

    @property
    def flux(self):
        return self._flux[self.mask]

    @property
    def sigma(self):
        return self._sigma[self.mask]


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
        Poisson noise (in f_lam). If not specified, will be unitary. Default is None
    masks : 1D or 2D array-like
        Mask to blot out bad pixels or emission regions. Must be castable to boolean. Default is None


    Attributes
    ----------
    waves : numpy.ndarray with self.shape
        The masked wavelength grid
    fluxes : numpy.ndarray with self.shape
        The masked flux grid
    sigmas : numpy.ndarray with self.shape
        The masked sigma grid

    Note
    ----
    If the waves, fluxes, and sigmas are provided as 1D arrays (say for a single order), they will be converted to 2D arrays with length 1 in the 0-axis.

    Warning
    -------
    For now, the Spectrum waves, fluxes, sigmas, and masks must be a rectangular grid. No ragged Echelle orders allowed.

    """

    def __init__(self, waves, fluxes, sigmas=None, masks=None, name="Spectrum"):
        self.waves = np.atleast_2d(waves)
        self.fluxes = np.atleast_2d(fluxes)

        if sigmas is not None:
            self.sigmas = np.atleast_2d(sigmas)
        else:
            self.sigmas = np.ones_like(self.fluxes)

        if masks is not None:
            self.masks = np.atleast_2d(masks)
        else:
            self.masks = np.ones_like(self.waves, dtype=bool)

        self._shape = self.waves.shape
        self.norders = self._shape[0]
        assert self.fluxes.shape == self._shape, "flux array incompatible shape."
        assert self.sigmas.shape == self._shape, "sigma array incompatible shape."
        assert self.masks.shape == self._shape, "mask array incompatible shape."
        self.orders = []
        for i in range(self.norders):
            self.orders.append(
                Order(self.waves[i], self.fluxes[i], self.sigmas[i], self.masks[i])
            )
        self.name = name

    def __getitem__(self, key: int):
        return self.orders[key]

    def __len__(self):
        return len(self.orders)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        if np.product(shape) != np.product(self._shape):
            raise ValueError(f"{shape} incompatible with shape {self.shape}")
        self.waves = self.waves.reshape(shape)
        self.fluxes = self.fluxes.reshape(shape)
        self.sigmas = self.sigmas.reshape(shape)
        self.masks = self.masks.reshape(shape)
        self._shape = shape
        self.orders = []
        for i in range(shape[0]):
            self.orders.append(
                Order(self.waves[i], self.fluxes[i], self.sigmas[i], self.masks[i])
            )

    def reshape(self, shape):
        waves = self.waves.reshape(shape)
        fluxes = self.fluxes.reshape(shape)
        sigmas = self.sigmas.reshape(shape)
        masks = self.masks.reshape(shape)
        return self.__class__(waves, fluxes, sigmas, masks, name=self.name)

    @property
    def masks(self):
        return self._masks

    @masks.setter
    def masks(self, masks):
        for o, m in zip(self.orders, masks):
            o.mask = m
        self._masks = masks

    @classmethod
    def load(cls, filename):
        """
        Load a spectrum from an hdf5 file

        Parameters
        ----------
        filename : str or path-like
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
        """

        with h5py.File(filename, "w") as base:
            base.create_dataset("waves", data=self.waves, compression=9)
            base.create_dataset("fluxes", data=self.fluxes, compression=9)
            base.create_dataset("sigmas", data=self.sigmas, compression=9)
            base.create_dataset("masks", data=self.masks, compression=9)
            if self.name is not None:
                base.attrs["name"] = self.name

    def __repr__(self):
        return f"{self.name} ({self.waves.shape[0]} orders)"
