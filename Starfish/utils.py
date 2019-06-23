import numpy as np

import Starfish.constants as C


def calculate_dv(wl):
    """
    Given a wavelength array, calculate the minimum ``dv`` of the array.

    :param wl: wavelength array
    :type wl: np.array

    :returns: (float) delta-v in units of km/s
    """
    return C.c_kms * np.min(np.diff(wl) / wl[:-1])


def calculate_dv_dict(wl_dict):
    """
    Given a ``wl_dict``, calculate the velocity spacing.

    :param wl_dict: wavelength dictionary
    :type wl_dict: dict

    :returns: (float) delta-v in units of km/s
    """
    CDELT1 = wl_dict["CDELT1"]
    dv = C.c_kms * (10 ** CDELT1 - 1)
    return dv


def create_log_lam_grid(dv, wl_start=3000.0, wl_end=13000.0):
    """
    Create a log lambda spaced grid with ``N_points`` equal to a power of 2 for
    ease of FFT.

    :param wl_start: starting wavelength (inclusive)
    :type wl_start: float, AA
    :param wl_end: ending wavelength (inclusive)
    :type wl_end: float, AA
    :param dv: upper bound on the size of the velocity spacing (in km/s)
    :type dv: float

    :returns: a wavelength dictionary containing the specified properties. Note
        that the returned dv will be <= specified dv.
    :rtype: wl_dict

    """
    assert wl_start < wl_end, "wl_start must be smaller than wl_end"

    CDELT_temp = np.log10(dv / C.c_kms + 1.0)
    CRVAL1 = np.log10(wl_start)
    CRVALN = np.log10(wl_end)
    N = (CRVALN - CRVAL1) / CDELT_temp
    NAXIS1 = 2
    while NAXIS1 < N:  # Make NAXIS1 an integer power of 2 for FFT purposes
        NAXIS1 *= 2

    CDELT1 = (CRVALN - CRVAL1) / (NAXIS1 - 1)

    p = np.arange(NAXIS1)
    wl = 10 ** (CRVAL1 + CDELT1 * p)
    return {"wl": wl, "CRVAL1": CRVAL1, "CDELT1": CDELT1, "NAXIS1": NAXIS1}
