import extinction
import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import j1

from Starfish.constants import c_kms
from Starfish.utils import calculate_dv


def resample(wave, flux, new_wave):
    # TODO docstring
    if np.any(new_wave <= 0):
        raise ValueError('Wavelengths must be positive')

    if flux.ndim > 1:
        interpolators = [InterpolatedUnivariateSpline(wave, fl, k=5) for fl in flux]
        return np.array([interpolator(new_wave) for interpolator in interpolators])
    else:
        return InterpolatedUnivariateSpline(wave, flux, k=5)(new_wave)


def instrumental_broaden(wave, flux, fwhm):
    # TODO docstring
    if fwhm < 0:
        raise ValueError('FWHM must be non-negative')
    dv = calculate_dv(wave)
    freq = np.fft.rfftfreq(flux.shape[-1], d=dv)
    flux_ff = np.fft.rfft(flux)

    sigma = fwhm / 2.355
    flux_ff *= np.exp(-2 * (np.pi * sigma * freq) ** 2)

    flux_final = np.fft.irfft(flux_ff, n=flux.shape[-1])
    return flux_final


def rotational_broaden(wave, flux, vsini):
    # TODO docstring
    if vsini <= 0:
        raise ValueError('vsini must be positive')

    dv = calculate_dv(wave)
    freq = np.fft.rfftfreq(flux.shape[-1], dv)
    flux_ff = np.fft.rfft(flux)
    # Calculate the stellar broadening kernel (Gray 2008)
    ub = 2. * np.pi * vsini * freq
    # Remove 0th frequency
    ub = ub[1:]
    sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
    flux_ff *= np.insert(sb, 0, 1.)
    flux_final = np.fft.irfft(flux_ff, n=flux.shape[-1])
    return flux_final


def doppler_shift(wave, vz):
    # TODO docstring
    dv = np.sqrt((c_kms + vz) / (c_kms - vz))
    return wave * dv


def extinct(wave, flux, Av, Rv=3.1, law='ccm89'):
    # TODO docstring
    if law not in ['ccm89', 'odonnell94', 'calzetti00', 'fitzpatrick99', 'fm07']:
        raise ValueError('Invalid extinction law given')
    if Av < 0:
        raise ValueError('Av must be positive')
    if Rv <= 0:
        raise ValueError('Rv must be positive')

    law_fn = eval('extinction.{}'.format(law))
    if law == 'fm07':
        A_l = law_fn(wave, Av)
    else:
        A_l = law_fn(wave, Av, Rv)
    flux_final = flux * 10 ** (-0.4 * A_l)
    return flux_final


def rescale(flux, w):
    # TODO docstring
    return flux * 10 ** w


def chebyshev_correct(wave, flux, coeffs):
    # TODO everything
    # have to scale wave to fit on domain [0, 1]
    if not isinstance(coeffs, np.ndarray):
        coeffs = np.array(coeffs)
    if coeffs.ndim == 1 and coeffs[0] != 1:
        raise ValueError('For single spectrum the linear Chebyshev coefficient (c[0]) must be 1')

    scale_wave = wave / wave.max()
    p = chebval(scale_wave, coeffs, tensor=False)
    return flux * p
