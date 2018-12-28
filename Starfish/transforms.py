
import numpy as np

class Transform:
    """
    This is the base transform class to be extended by its constituents
    """
    def __call__(self, wave, flux):
        if not isinstance(wave, np.ndarray):
            wave = np.array(wave)
        if not isinstance(flux, np.ndarray):
            flux = np.array(flux)
        return self._transform(wave, flux)

    def _transform(self, wave, flux):
        """
        This is the method to be overwritten by inheriting members. It must contain
        the same callsign as this method

        :param wave: the wavelength array in Angstrom
        :type wave: numpy.ndarray
        :param flux: the flux array in erg/cm^2/s/A
        :type flux: numpy.ndarray

        :return: tuple of (wave, flux) of the transformed data
        """
        raise NotImplementedError('Must be implemented by subclasses of Transform')


class Truncate(Transform):
    """
    This class truncates a spectra to a given wavelength range with an optional
    buffer on each side.

    :param wl_range: The desired wavelength range in Angstrom. If None, will have
        no truncation (0, numpy.inf). Default is None
    :type wl_range: tuple of (min, max)
    :param buffer: The desired buffer in Angstrom. Default is 0
    :type buffer: float
    """

    def __init__(self, wl_range=None, buffer=0):
        if wl_range is None:
            self.wl_range = 0, np.inf
            buffer = 0
        else:
            self.wl_range = wl_range
        self.min = self.wl_range[0]
        self.max = self.wl_range[1]
        self.buffer = buffer

    def _transform(self, wave, flux):
        return truncate(wave, flux, self.wl_range, self.buffer)

def truncate(wave, flux, wl_range, buffer):
    wl_min, wl_max = wl_range
    wl_min -= buffer
    wl_max += buffer
    mask = (wave >= wl_min) & (wave <= wl_max)
    return wave[mask], flux[mask]


class Convolve(Transform):
    pass