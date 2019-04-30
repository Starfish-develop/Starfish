import extinction
import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import j1

from Starfish.constants import c_kms
from Starfish.utils import calculate_dv


def resample(wave, flux, new_wave):
    """
    Resample onto a new wavelength grid using k=5 spline interpolation

    Parameters
    ----------
    wave : array_like
        The original wavelength grid
    flux : array_like
        The fluxes to resample
    new_wave : array_like
        The new wavelength grid

    Raises
    ------
    ValueError
        If the new wavelength grid is not strictly increasing monotonic

    Returns
    -------
    numpy.ndarray
        The resampled flux with the same 1st dimension as the input flux
    """

    if np.any(new_wave <= 0):
        raise ValueError('Wavelengths must be positive')

    if flux.ndim > 1:
        interpolators = [InterpolatedUnivariateSpline(
            wave, fl, k=5) for fl in flux]
        return np.array([interpolator(new_wave) for interpolator in interpolators])
    else:
        return InterpolatedUnivariateSpline(wave, flux, k=5)(new_wave)


def instrumental_broaden(wave, flux, fwhm):
    """
    Broadens given flux by convolving with a Gaussian kernel appropriate for a spectrograph's instrumental properties. Follows the given equation

    .. math::
        f = f * \\mathcal{F}^{\\text{inst}}_v

    .. math::
        \\mathcal{F}^{\\text{inst}}_v = \\frac{1}{\\sqrt{2\\pi \\sigma^2}} \\exp \\left[-\\frac12 \\left( \\frac{v}{\\sigma} \\right)^2 \\right]

    This is carried out by multiplication in the Fourier domain rather than using a convolution function.

    Parameters
    ----------
    wave : array_like
        The current wavelength grid
    flux : array_like
        The current flux
    fwhm : float
        The full width half-maximum of the instrument in km/s. Note that this is equivalent to :math:`2.355\\cdot \\sigma`

    Raises
    ------
    ValueError
        If the full width half maximum is negative.

    Returns
    -------
    numpy.ndarray
        The broadened flux with the same shape as the input flux
    """

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
    """
    Broadens flux according to a rotational broadening kernel from Gray (2005) [1]_

    Parameters
    ----------
    wave : array_like
        The current wavelength grid
    flux : array_like
        The current flux
    vsini : float
        The rotational velocity in km/s

    Raises
    ------
    ValueError
        if `vsini` is not positive

    Returns
    -------
    numpy.ndarray
        The broadened flux with the same shape as the input flux


    .. [1] Gray, D. (2005). *The observation and Analysis of Stellar Photospheres*. Cambridge: Cambridge University Press. doi:10.1017/CB09781316036570
    """

    if vsini <= 0:
        raise ValueError('vsini must be positive')

    dv = calculate_dv(wave)
    freq = np.fft.rfftfreq(flux.shape[-1], dv)
    flux_ff = np.fft.rfft(flux)
    # Calculate the stellar broadening kernel (Gray 2008)
    ub = 2. * np.pi * vsini * freq
    # Remove 0th frequency
    ub = ub[1:]
    sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + \
        3. * np.sin(ub) / (2 * ub ** 3)
    flux_ff *= np.insert(sb, 0, 1.)
    flux_final = np.fft.irfft(flux_ff, n=flux.shape[-1])
    return flux_final


def doppler_shift(wave, vz):
    """
    Doppler shift a spectrum

    Parameters
    ----------
    wave : array_like
        The unshifted wavelengths
    vz : float
        The doppler velocity in km/s

    Returns
    -------
    numpy.ndarray
        Altered wavelengths with the same shape as the input wavelengths
    """

    dv = np.sqrt((c_kms + vz) / (c_kms - vz))
    return wave * dv


def extinct(wave, flux, Av, Rv=3.1, law='ccm89'):
    """
    Extinct a spectrum following one of many empirical extinction laws. This makes use of the `extinction` package.

    Parameters
    ----------
    wave : array_like
        The input wavelengths in Angstrom
    flux : array_like
        The input fluxes
    Av : float
        The absolute attenuation
    Rv : float, optional
        The relative attenuation (the default is 3.1, which is the Milky Way average)
    law : str, optional
        The extinction law to use. One of `{'ccm89', 'odonnell94', 'calzetti00', 'fitzpatrick99', 'fm07'}` (the default is 'ccm89')

    Raises
    ------
    ValueError
        If `law` does not match one of the availabe laws
    ValueError
        If Av or Rv is not positive

    Returns
    -------
    numpy.ndarray
        The extincted fluxes, with same shape as input fluxes.
    """

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


def rescale(flux, log_scale):
    """
    Rescale the given flux via the following equation

    .. math:: f(\\log \\Omega) = f \\times 10^{\\log \\Omega}

    Parameters
    ----------
    flux : array_like
        The input fluxes
    log_scale : float
        The base-10 logarithm of the scaling factor

    Returns
    -------
    numpy.ndarray
        The rescaled fluxes with the same shape as the input fluxes
    """

    return flux * (10 ** log_scale)


def chebyshev_correct(wave, flux, coeffs):
    # TODO everything
    # have to scale wave to fit on domain [0, 1]
    if not isinstance(coeffs, np.ndarray):
        coeffs = np.array(coeffs)
    if coeffs.ndim == 1 and coeffs[0] != 1:
        raise ValueError(
            'For single spectrum the linear Chebyshev coefficient (c[0]) must be 1')

    scale_wave = wave / wave.max()
    p = chebval(scale_wave, coeffs, tensor=False)
    return flux * p
