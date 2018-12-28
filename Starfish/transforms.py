
import numpy as np
from scipy.special import j1
from scipy.interpolate import InterpolatedUnivariateSpline

import Starfish.constants as C
from Starfish.grid_tools import Instrument
from Starfish.utils import create_log_lam_grid, calculate_dv_dict, calculate_dv

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


class FTransform:
    """
    This is the base transform class for transforms that operate in Fourier space
    like convolution and resampling
    """
    def __call__(self, fwave, flux):
        """
        :param fwave: the wavelength array in the Fourier domain
        :type fwave: numpy.ndarray
        :param flux: the flux array in erg/cm^2/s/A
        :type flux: numpy.ndarray

        :return: tuple of (wave, flux) of the transformed data
        """
        if not isinstance(fwave, np.ndarray):
            wave = np.array(fwave)
        if not isinstance(flux, np.ndarray):
            flux = np.array(flux)
        return self._transform(fwave, flux)

    def _transform(self, fwave, flux):
        """
        This is the method to be overwritten by inheriting members. It must contain
        the same callsign as this method

        :param fwave: the wavelength array in the Fourier domain
        :type fwave: numpy.ndarray
        :param flux: the flux array in erg/cm^2/s/A
        :type flux: numpy.ndarray

        :return: tuple of (logwave, flux) of the transformed data
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
        wl_min = self.min - self.buffer
        wl_max = self.max + self.buffer
        mask = (wave >= wl_min) & (wave <= wl_max)
        return wave[mask], flux[mask]

def truncate(wave, flux, wl_range, buffer):
    t = Truncate(wl_range, buffer)
    return t(wave, flux)


class Broaden(FTransform):
    """
    Not sure if this is correct
    """

    def __init__(self, inst=None, vz=None, vsini=None):
        if isinstance(inst, Instrument):
            self.inst = inst.FWHM
        else:
            self.inst = inst
        self.vz = vz
        self.vsini = vsini

    def _transform(self, fwave, flux):
        # Check for short-circuit
        if self.inst is None and self.vz is None and self.vsini is None:
            return fwave, flux

        if self.vz is not None:
           fwave *= np.sqrt((C.c_kms + self.vz) / (C.c_kms - self.vz))
        log_dv = calculate_dv(fwave)
        ss = np.fft.rfftfreq(len(fwave), d=log_dv)
        sigma = 0

        taper = 1.
        if self.inst is not None:
            taper = np.exp(-2 * (np.pi ** 2) * ((self.inst/2.35) ** 2) * (ss ** 2))
            sigma += self.inst

        if self.vsini is not None:
            # Calculate the stellar broadening kernel
            ub = 2. * np.pi * self.vsini * ss
            sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
            # set zeroth frequency to 1 separately (DC term)
            sb[0] = 1.
            taper = sb * taper
            sigma += self.vsini

        FF = np.fft.rfft(flux, len(fwave))
        FF_tap = FF * taper
        # do IFFT
        fl_final = np.fft.irfft(FF_tap, len(flux))

        return fwave, flux

def broaden(fwave, flux, vinst, vz, vsini):
    t = Broaden(vinst, vz, vsini)
    return t(fwave, flux)

class Resample(FTransform):
    """
    I don't think this is correct.
    """

    def __init__(self, wave):
        self.wave=wave

    def _transform(self, fwave, flux):

        interp = InterpolatedUnivariateSpline(fwave, flux)
        flux_final = interp(self.wave)
        return self.wave, flux_final