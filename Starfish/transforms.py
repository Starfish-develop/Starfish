import extinction
import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import j1

import Starfish.constants as C
from Starfish.grid_tools.instruments import Instrument
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
    like convolution and resampling. When called using ``__call__`` it will take in normal flux and
    wavelength components and return normal flux and wavelength components. When called using
    ``transform`` it will take Fourier wavelength and fluxes in and return Fourier wavelength and fluxes.
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
    """
    This special class does nothing to the input data. Its primary use is for
    throwing into transformation chains that expect a transform based on
    some input, when that input isn't always defined.
    """
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
    :type inst: float or :class:`Instrument`

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

        \\lambda = \\lambda_0 \\sqrt{ \\frac{c + v_z}{c - v_z} }

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

class CalibrationCorrect(Transform):
    """
    Uses Chebyshev polynomials to correct for flux-calibration errors.

    :param c: Chebyshev coefficients.
    :type c: array_like
    """

    def __init__(self, c):
        if not isinstance(c, np.ndarray):
            c = np.array(c)
        self.c = c

    def transform(self, wave, flux):
        xs = np.arange(len(wave))
        return wave, flux * chebval(xs, self.c)

def calibration_correct(wave, flux, c):
    """Helper function for :class:`CalibrationCorrect`"""
    t = CalibrationCorrect(c)
    return t(wave, flux)

class Extinct(Transform):
    """
    Extincts a given spectra according to the given law. Uses the ``extinction``
    package under the hood

    :param law: the extinction law, one of {'ccm89', 'odonnell94', 'calzetti00',
        'fitzpatrick99', 'fm07'}
    :type law: str
    :param Av: The scaling total extinction value.
    :type Av: float
    :param Rv: The ratio of total to selective extinction. If using law 'fm07' you do
        not need to provide this (fixed 3.1).
    :type Rv: float

    :raises ValueError: If not using an expected law or ill-specifying Av or Rv
    """
    LAWS = {
        'ccm89': extinction.ccm89,
        'odonnell94': extinction.odonnell94,
        'calzetti00': extinction.calzetti00,
        'fitzpatrick99': extinction.fitzpatrick99,
        'fm07': extinction.fm07,
    }
    def __init__(self, law, Av, Rv=None):
        if not law in self.LAWS:
            raise ValueError('Need to specify a law from {}'.format(self.LAWS.keys()))
        if Av < 0:
            raise ValueError('Cannot have negative extinction')
        if Rv is None or Rv < 0 and law is not 'fm07':
            raise ValueError('Must provide positive r_v for law "{}"'.format(law))
        elif law is 'fm07':
            Rv = None
        self.law = self.LAWS[law]
        self.Av = Av
        self.Rv = Rv

    def transform(self, wave, flux):
        if self.Rv is not None:
            extinct_mag = self.law(wave, self.Av, self.Rv)
        else:
            extinct_mag = self.law(wave, self.Av)
        extinct_flux = extinction.apply(extinct_mag, flux)
        return wave, extinct_flux


def extinct(wave, flux, law, Av, Rv):
    t = Extinct(law, Av, Rv)
    return t(wave, flux)

class Scale(Transform):
    """
    Scales the flux by the given scaling factor :math:`\\log \\Omega`. This performs the operation

    .. math::

        F = F_0 10^{\\log \\Omega}

    :param logOmega: The base-10 logarithm of the scaling factor. This can represent
        the logarithm of the distance to an object or 2.5 times a magnitude scaling
        factor.
    :type logOmega: float
    """

    def __init__(self, logOmega):
        self.logOmega = logOmega

    def transform(self, wave, flux):
        return wave, flux * 10 ** self.logOmega

def scale(wave, flux, logOmega):
    t = Scale(logOmega)
    return t(wave, flux)

