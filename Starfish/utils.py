from typing import Sequence

import numpy as np

import Starfish.constants as C


def calculate_dv(wave: Sequence):
    """
    Given a wavelength array, calculate the minimum ``dv`` of the array.

    Parameters
    ----------
    wave : array-like
        The wavelength array

    Returns
    -------
    float
        delta-v in units of km/s
    """
    return C.c_kms * np.min(np.diff(wave) / wave[:-1])


def calculate_dv_dict(wave_dict):
    """
    Given a ``wave_dict``, calculate the velocity spacing.

    Parameters
    ---------
    wave_dict : dict
        wavelength dictionary

    Returns
    -------
    float
        delta-v in units of km/s
    """
    CDELT1 = wave_dict["CDELT1"]
    dv = C.c_kms * (10 ** CDELT1 - 1)
    return dv


def create_log_lam_grid(dv, start, end):
    """
    Create a log lambda spaced grid with ``N_points`` equal to a power of 2 for
    ease of FFT.

    Parameters
    ----------
    dv : float
        Upper bound on the velocity spacing in km/s
    start : float
        starting wavelength (inclusive) in Angstrom
    end : float
        ending wavelength (inclusive) in Angstrom

    Returns
    -------
    dict
        a wavelength dictionary containing the specified properties. Note that the returned dv will be less than or equal to the specified dv.

    Raises
    ------
    ValueError
        If starting wavelength is not less than ending wavelength
    ValueError
        If any of the wavelengths are less than 0
    """
    if start >= end:
        raise ValueError("Wavelength must be increasing, but start >= end")

    if start <= 0 or end <= 0:
        raise ValueError("Cannot have negative or 0 wavelength")

    CDELT_temp = np.log10(dv / C.c_kms + 1.0)
    CRVAL1 = np.log10(start)
    CRVALN = np.log10(end)
    N = (CRVALN - CRVAL1) / CDELT_temp
    NAXIS1 = 2
    while NAXIS1 < N:  # Make NAXIS1 an integer power of 2 for FFT purposes
        NAXIS1 *= 2

    CDELT1 = (CRVALN - CRVAL1) / (NAXIS1 - 1)

    p = np.arange(NAXIS1)
    wl = 10 ** (CRVAL1 + CDELT1 * p)
    return {"wl": wl, "CRVAL1": CRVAL1, "CDELT1": CDELT1, "NAXIS1": NAXIS1}
