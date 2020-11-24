import extinction  # This may be marked as unused, but is necessary
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
        raise ValueError("Wavelengths must be positive")

    if flux.ndim > 1:
        interpolators = [InterpolatedUnivariateSpline(wave, fl, k=5) for fl in flux]
        return np.array([interpolator(new_wave) for interpolator in interpolators])
    else:
        return InterpolatedUnivariateSpline(wave, flux, k=5)(new_wave)


def instrumental_broaden(wave, flux, fwhm):
    """
    Broadens given flux by convolving with a Gaussian kernel appropriate for a
    spectrograph's instrumental properties. Follows the given equation

    .. math::
        f = f * \\mathcal{F}^{\\text{inst}}_v

    .. math::
        \\mathcal{F}^{\\text{inst}}_v = \\frac{1}{\\sqrt{2\\pi \\sigma^2}} \\exp \\left[-\\frac12 \\left( \\frac{v}{\\sigma} \\right)^2 \\right]

    This is carried out by multiplication in the Fourier domain rather than using a
    convolution function.

    Parameters
    ----------
    wave : array_like
        The current wavelength grid
    flux : array_like
        The current flux
    fwhm : float
        The full width half-maximum of the instrument in km/s. Note that this is
        quivalent to :math:`2.355\\cdot \\sigma`

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
        raise ValueError("FWHM must be non-negative")
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


    .. [1] Gray, D. (2005). *The observation and Analysis of Stellar Photospheres*.
    Cambridge: Cambridge University Press. doi:10.1017/CB09781316036570
    """

    if vsini <= 0:
        raise ValueError("vsini must be positive")

    dv = calculate_dv(wave)
    freq = np.fft.rfftfreq(flux.shape[-1], dv)
    flux_ff = np.fft.rfft(flux)
    # Calculate the stellar broadening kernel (Gray 2008)
    ub = 2.0 * np.pi * vsini * freq
    # Remove 0th frequency
    ub = ub[1:]
    sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3.0 * np.sin(ub) / (2 * ub ** 3)
    flux_ff *= np.insert(sb, 0, 1.0)
    flux_final = np.fft.irfft(flux_ff, n=flux.shape[-1])
    return flux_final


def doppler_shift(wave, vz):
    """
    Doppler shift a spectrum according to the formula

    .. math::
        \\lambda \\cdot \\sqrt{\\frac{c + v_z}{c - v_z}}

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


def extinct(wave, flux, Av, Rv=3.1, law="ccm89"):
    """
    Extinct a spectrum following one of many empirical extinction laws. This makes use
    of the `extinction` package. In general, it follows the form

    .. math:: f \\cdot 10^{-0.4 A_V \\cdot A_\\lambda(R_V)}

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
        The extinction law to use. One of `{'ccm89', 'odonnell94', 'calzetti00',
        'fitzpatrick99', 'fm07'}` (the default is 'ccm89')

    Raises
    ------
    ValueError
        If `law` does not match one of the availabe laws
    ValueError
        If Rv is not positive

    Returns
    -------
    numpy.ndarray
        The extincted fluxes, with same shape as input fluxes.
    """

    if law not in ["ccm89", "odonnell94", "calzetti00", "fitzpatrick99", "fm07"]:
        raise ValueError("Invalid extinction law given")
    if Rv <= 0:
        raise ValueError("Rv must be positive")

    law_fn = eval("extinction.{}".format(law))
    if law == "fm07":
        A_l = law_fn(wave.astype(np.double), Av)
    else:
        A_l = law_fn(wave.astype(np.double), Av, Rv)
    flux_final = flux * 10 ** (-0.4 * A_l)
    return flux_final


def rescale(flux, scale):
    """
    Rescale the given flux via the following equation

    .. math:: f \\cdot \\Omega

    Parameters
    ----------
    flux : array_like
        The input fluxes
    scale : float or array_like
        The scaling factor. If an array, must have same shape as the batch dimension of
        :attr:`flux`

    Returns
    -------
    numpy.ndarray
        The rescaled fluxes with the same shape as the input fluxes
    """
    scale = np.atleast_1d(scale)
    if len(scale) > 1:
        scale = scale[:, np.newaxis]
    return flux * scale


def renorm(wave, flux, reference_flux):
    """
    Renormalize one spectrum to another

    This uses the :meth:`rescale` function with a :attr:`log_scale` of

    .. math::

        \\log \\Omega = \\left. \\int{f^{*}(w) dw} \\middle/ \\int{f(w) dw} \\right.

    where :math:`f^{*}` is the reference flux, :math:`f` is the source flux, and the
    integrals are over a common wavelength grid

    Parameters
    ----------
    wave : array_like
        The wavelength grid for the source flux
    flux : array_like
        The flux for the source
    reference_flux : array_like
        The reference source to renormalize to

    Returns
    -------
    numpy.ndarray
        The renormalized flux
    """
    factor = _get_renorm_factor(wave, flux, reference_flux)
    return rescale(flux, factor)


def _get_renorm_factor(wave, flux, reference_flux):
    ref_int = np.trapz(reference_flux, wave)
    flux_int = np.trapz(flux, wave, axis=-1)
    return ref_int / flux_int


def chebyshev_correct(wave, flux, coeffs):
    """
    Multiply the input flux by a Chebyshev series in order to correct for
    calibration-level discrepancies.

    Parameters
    ----------
    wave : array-lioke
        Input wavelengths
    flux : array-like
        Input flux
    coeffs : array-like
        The coefficients for the chebyshev series.

    Returns
    -------
    numpy.ndarray
        The corrected flux

    Raises
    ------
    ValueError
        If only processing a single spectrum and the linear coefficient is not 1.
    """
    # have to scale wave to fit on domain [0, 1]
    coeffs = np.asarray(coeffs)
    if coeffs.ndim == 1 and coeffs[0] != 1:
        raise ValueError(
            "For single spectrum the linear Chebyshev coefficient (c[0]) must be 1"
        )

    scale_wave = wave / wave.max()
    p = chebval(scale_wave, coeffs, tensor=False)
    return flux * p
