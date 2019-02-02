import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import j1

from Starfish.constants import c_kms
from Starfish.utils import create_log_lam_grid, calculate_dv


def resample(wave, flux, new_wave):
    interpolators = [InterpolatedUnivariateSpline(wave, fl, k=5) for fl in flux]
    return np.array([interpolator(new_wave) for interpolator in interpolators])


def instrumental_broaden(wave, flux, fwhm):
    pass


def rfffreq(n, d=1.0):
    N = n // 2 + 1
    f = tt.arange(0, N, dtype=int)
    return f / (n * d)


def rotational_broaden(wave, flux, vsini):
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
    dv = np.sqrt((c_kms + vz) / (c_kms - vz))
    return wave * dv


def extinct(wave, flux, Av, Rv=3.1, law='ccm89'):
    law = eval(f'extinction.{law}')
    A_l = law(wave, Av, Rv)
    flux_final = flux * 10 ** (-0.4 * A_l)
    return flux_final


def rescale(flux, w):
    # Should work with theano standard
    return flux * 10 ** w


def chebyshev_correct(wave, flux, coeffs):
    pass
