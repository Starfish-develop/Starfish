import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import j1

import Starfish.constants as C
from Starfish.grid_tools import Instrument
from Starfish.utils import calculate_dv


class Transform:
    """
    This is the base transform class to be extended by its constituents
    """

    def __call__(self, wave, flux):
        """
        :param wave: the wavelength array in Angstrom
        :type wave: iterable
        :param flux: the flux_ff array in erg/cm^2/s/AA
        :type flux: iterable

        :return: tuple of (wave, flux) of the transformed data
        """
        if not isinstance(wave, np.ndarray):
            wave = np.array(wave)
        if not isinstance(flux, np.ndarray):
            flux = np.array(flux)
        return self.transform(wave, flux)

    def transform(self, wave, flux):
        """
        This is the method to be overwritten by inheriting members. It must contain
        the same callsign as this method

        :param wave: the wavelength array in Angstrom
        :type wave: numpy.ndarray
        :param flux: the flux array in erg/cm^2/s/AA
        :type flux: numpy.ndarray

        :return: tuple of (wave, flux) of the transformed data
        """
        raise NotImplementedError('Must be implemented by subclasses of Transform')


class FTransform:
    """
    This is the base transform class for transforms that operate in Fourier space
    like convolution and resampling. When called using `__call__` it will take in normal flux and
    wavelength components and return normal flux and wavelength components. When called using
    `transform` it will take Fourier wavelength and fluxes in and return Fourier wavelength and fluxes.
    """

    def __call__(self, wave, flux):
        """
        :param wave: the wavelength array in AA
        :type wave: iterable
        :param flux: the flux_ff array in erg/cm^2/s/AA
        :type flux: iterable

        :return: tuple of (wave, flux) of the transformed data
        """
        if not isinstance(wave, np.ndarray):
            wave = np.array(wave)
        if not isinstance(flux, np.ndarray):
            flux = np.array(flux)

        # Convert to Fourier domain, maintain doppler content
        dv = calculate_dv(wave)
        wave_ff = np.fft.rfftfreq(len(wave), d=dv)
        flux_ff = np.fft.rfft(flux)

        wave_ff_t, flux_ff_t = self.transform(wave_ff, flux_ff)
        wave_final = wave
        flux_final = np.fft.irfft(flux_ff_t)
        return wave_final, flux_final

    def transform(self, wave_ff, flux_ff):
        """
        This is the method to be overwritten by inheriting members. It must contain
        the same callsign as this method

        :param wave_ff: the wavelength array in the Fourier domain
        :type wave_ff: numpy.ndarray
        :param flux_ff: the flux array in the Fourier domain
        :type flux_ff: numpy.ndarray

        :return: tuple of (wave_ff, flux_ff) of the transformed data
        """
        raise NotImplementedError('Must be implemented by subclasses of Transform')

class NullTransform(Transform):
    def transform(self, wave, flux):
        return wave, flux

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

    def transform(self, wave, flux):
        wl_min = self.min - self.buffer
        wl_max = self.max + self.buffer
        mask = (wave >= wl_min) & (wave <= wl_max)
        return wave[mask], flux[mask]


def truncate(wave, flux, wl_range, buffer):
    """Helper function for :class:`Truncate`"""
    t = Truncate(wl_range, buffer)
    return t(wave, flux)


class InstrumentalBroaden(FTransform):
    """
    This class will provide the kernel transformation for instrumental broadening
     in the Fourier domain.

    :param inst: The instrumental velocity FWHM in km/s.
    :type inst: float or :class:`Starfish.grid_tools.Instrument`

    :raises ValueError: If the instrumental FWHM is less than 0
    """

    def __init__(self, inst):
        if isinstance(inst, Instrument):
            self.inst = inst.FWHM
        else:
            self.inst = inst

        if self.inst < 0.0:
            raise ValueError("Cannot have a negative instrumental velocity")

    def transform(self, wave_ff, flux_ff):
        # Convert from FWHM to standard deviation for Gaussian
        sigma = (self.inst / 2.355)
        # Multiply by the equivalent of the Fourier transform of a Gaussian
        flux_ff *= np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (wave_ff ** 2))
        return wave_ff, flux_ff


class RotationalBroaden(FTransform):
    """
    This class will provide the kernel transformation for
    rotational broadening in the Fourier domain.

    :param vsini: The rotational velocity in km/s.
    :type vsini: float

    :raises ValueError: If vsini is not positive
    """

    def __init__(self, vsini):
        if not vsini > 0:
            raise ValueError('Must have a positive rotational velocity.')

        self.vsini = vsini

    def transform(self, wave_ff, flux_ff):
        # Calculate the stellar broadening kernel (Gray 2008)
        ub = 2. * np.pi * self.vsini * wave_ff
        # Artifically push to avoid divde-by-Zero
        ub[0] = np.finfo(np.float16).tiny
        sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
        # set zeroth frequency to 1 separately (DC term)
        sb[0] = 1.
        flux_ff *= sb

        return wave_ff, flux_ff


def instrumental_broaden(wave, flux, vinst):
    """Helper function for :class:`InstrumentalBroaden`"""
    t = InstrumentalBroaden(vinst)
    return t(wave, flux)


def rotational_broaden(wave, flux, vsini):
    """Helper function for :class:`RotationalBroaden`"""
    t = RotationalBroaden(vsini)
    return t(wave, flux)


class DopplerShift(Transform):
    """
    This class will doppler shift given data by the equation

    .. math::
        \lambda = \lambda_0 \sqrt{\frac{c + v_z}{c - v_z}}

    where :math:`c` and :math:`v_z` are given in similar units- in our case we
    choose :math:`km/s`.

    :param vz: the doppler velocity in km/s. Positive implies redshift.
    :type vz: float
    """

    def __init__(self, vz):
        self.vz = vz

    def transform(self, wave, flux):
        wave_final = wave * np.sqrt((C.c_kms + self.vz) / (C.c_kms - self.vz))
        return wave_final, flux


def doppler_shift(wave, flux, vz):
    """Helper function for :class:`DopplerShift`"""
    t = DopplerShift(vz)
    return t(wave, flux)


class Resample(Transform):
    """
    Resamples the given data using a 5-spline interpolation scheme.

    :param new_wave: The wavelengths to interpolate to, in Angstrom
    :type new_wave: iterable

    :raises ValueError: If wavelengths are not positive, non-zero.
    """

    def __init__(self, new_wave):
        if not isinstance(new_wave, np.ndarray):
            new_wave = np.array(new_wave)
        if not np.all(new_wave > 0):
            raise ValueError('Must provide positive, non-zero wavelengths')
        self.wave_final = new_wave

    def transform(self, wave, flux):
        interp = InterpolatedUnivariateSpline(wave, flux)
        flux_final = interp(self.wave_final)
        return self.wave_final, flux_final


def resample(wave, flux, new_wave):
    """Helper function for :class:`Resample`"""
    t = Resample(new_wave)
    return t(wave, flux)
