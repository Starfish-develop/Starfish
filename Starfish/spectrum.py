import h5py
import numpy as np


class DataSpectrum:
    """
    Object to store astronomical spectra.

    Parameters
    ----------
    wls : 1D or 2D array-like
        wavelength in Angtsrom
    fls : 1D or 2D array-like
         flux (in f_lam)
    stds : 1D or 2D array-like, optional
        Poisson noise (in f_lam). If not specified, will be unitary. Default is None
    masks : 1D or 2D array-like
        Mask to blot out bad pixels or emission regions. Must be castable to boolean. Default is None

    .. note::
        If the wl, fl, are provided as 1D arrays (say for a single order), they will be converted to 2D arrays with length 1
        in the 0-axis.

    Properties
    ----------
    waves : numpy.ndarray with self.shape
        The masked wavelength grid
    fluxes : numpy.ndarray with self.shape
        The masked flux grid
    sigmas : numpy.ndarray with self.shape
        The masked sigma grid

    .. warning::
       For now, the DataSpectrum wls, fls, sigmas, and masks must be a rectangular grid. No ragged Echelle orders allowed.

    """

    def __init__(self, wls, fls, stds=None, masks=None, orders='all', name=None):
        self.wls = np.atleast_2d(wls)
        self.fls = np.atleast_2d(fls)
        if stds is not None:
            self.stds = np.atleast_2d(stds)
        else:
            self.stds = np.ones_like(self.fls)

        if masks is not None:
            self.masks = np.atleast_2d(masks)
        else:
            self.masks = np.ones_like(self.wls, dtype='b')

        self.shape = self.wls.shape
        assert self.fls.shape == self.shape, 'flux array incompatible shape.'
        assert self.stds.shape == self.shape, 'sigma array incompatible shape.'
        assert self.masks.shape == self.shape, 'mask array incompatible shape.'

        if orders != 'all':
            # can either be a numpy array or a list
            orders = np.array(orders)  # just to make sure
            self.wls = self.wls[orders]
            self.fls = self.fls[orders]
            self.stds = self.stds[orders]
            self.masks = self.masks[orders]
            self.shape = self.wls.shape
            self.orders = orders
        else:
            self.orders = np.arange(self.shape[0])

        self.name = name

    @property
    def waves(self):
        waves = self.wls[self.masks]
        return waves

    @property
    def fluxes(self):
        fluxes = self.fls[self.masks]
        return fluxes

    @property
    def sigmas(self):
        sigmas = self.stds[self.masks]
        return sigmas

    @classmethod
    def load(cls, filename):
        """
        Load a spectrum from an hdf5 file

        Parameters
        ----------
        filename : str or path-like
        """
        with h5py.File(filename, 'r') as base:
            wls = base['wls'][:]
            fls = base['fls'][:]
            sigmas = base['sigmas'][:]
            masks = base['masks'][:]
        return cls(wls, fls, sigmas, masks)

    def save(self, filename):
        """
        Takes the current DataSpectrum and writes it to an HDF5 file.

        Parameters
        ----------
        filename: str or path-like
            The filename to write to. Will not create any missing directories.
        """

        with h5py.File(filename, 'w') as base:
            base.create_dataset('wls', data=self.wls, compression=9)
            base.create_dataset('fls', data=self.fls, compression=9)
            base.create_dataset('sigmas', data=self.stds, compression=9)
            base.create_dataset('masks', data=self.masks, compression=9)
