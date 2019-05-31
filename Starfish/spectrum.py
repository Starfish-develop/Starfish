import h5py
import numpy as np

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
    For now, the DataSpectrum waves, fluxes, sigmas, and masks must be a rectangular grid. No ragged Echelle orders allowed.

    """

    def __init__(self, waves, fluxes, sigmas=None, masks=None, name='Spectrum'):
        self._waves = np.atleast_2d(waves)
        self._fluxes = np.atleast_2d(fluxes)

        if sigmas is not None:
            self._sigmas = np.atleast_2d(sigmas)
        else:
            self._sigmas = np.ones_like(self._fluxes)

        if masks is not None:
            self.masks = np.atleast_2d(masks)
        else:
            self.masks = np.ones_like(self._waves, dtype=bool)

        self.shape = self._waves.shape
        self.norders = self.shape[0]
        assert self._fluxes.shape == self.shape, 'flux array incompatible shape.'
        assert self._sigmas.shape == self.shape, 'sigma array incompatible shape.'
        assert self.masks.shape == self.shape, 'mask array incompatible shape.'

        self.name = name

    @property
    def waves(self):
        return np.squeeze(self._waves[self.masks].reshape(self.norders, -1))

    @property
    def fluxes(self):
        return np.squeeze(self._fluxes[self.masks].reshape(self.norders, -1))

    @property
    def sigmas(self):
        return np.squeeze(self._sigmas[self.masks].reshape(self.norders, -1))

    @classmethod
    def load(cls, filename):
        """
        Load a spectrum from an hdf5 file

        Parameters
        ----------
        filename : str or path-like
        """
        with h5py.File(filename, 'r') as base:
            if 'name' in base.attrs:
                name = base.attrs['name']
            else:
                name = None
            waves = base['waves'][:]
            fluxes = base['fluxes'][:]
            sigmas = base['sigmas'][:]
            masks = base['masks'][:]
        return cls(waves, fluxes, sigmas, masks, name=name)

    def save(self, filename):
        """
        Takes the current DataSpectrum and writes it to an HDF5 file.

        Parameters
        ----------
        filename: str or path-like
            The filename to write to. Will not create any missing directories.
        """

        with h5py.File(filename, 'w') as base:
            base.create_dataset('waves', data=self._waves, compression=9)
            base.create_dataset('fluxes', data=self._fluxes, compression=9)
            base.create_dataset('sigmas', data=self._sigmas, compression=9)
            base.create_dataset('masks', data=self.masks, compression=9)
            if self.name is not None:
                base.attrs['name'] = self.name

    def __repr__(self):
        return f'{self.name} ({self._waves.shape[0]} orders)'
