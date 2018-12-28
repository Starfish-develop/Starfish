import sys

import h5py
import numpy as np
from astropy.io import ascii
from astropy.table import Table

from Starfish import constants as C

def gelman_rubin(samplelist):
    '''
    Given a list of flatchains from separate runs (that already have burn in cut
    and have been trimmed, if desired), compute the Gelman-Rubin statistics in
    Bayesian Data Analysis 3, pg 284. If you want to compute this for fewer
    parameters, then slice the list before feeding it in.
    '''

    full_iterations = len(samplelist[0])
    assert full_iterations % 2 == 0, "Number of iterations must be even. Try cutting off a different number of burn in samples."
    shape = samplelist[0].shape
    # make sure all the chains have the same number of iterations
    for flatchain in samplelist:
        assert len(flatchain) == full_iterations, "Not all chains have the same number of iterations!"
        assert flatchain.shape == shape, "Not all flatchains have the same shape!"

    # make sure all chains have the same number of parameters.

    # Following Gelman,
    # n = length of split chains
    # i = index of iteration in chain
    # m = number of split chains
    # j = index of which chain
    n = full_iterations // 2
    m = 2 * len(samplelist)
    nparams = samplelist[0].shape[-1]  # the trailing dimension of a flatchain

    # Block the chains up into a 3D array
    chains = np.empty((n, m, nparams))
    for k, flatchain in enumerate(samplelist):
        chains[:, 2 * k, :] = flatchain[:n]  # first half of chain
        chains[:, 2 * k + 1, :] = flatchain[n:]  # second half of chain

    # Now compute statistics
    # average value of each chain
    avg_phi_j = np.mean(chains, axis=0, dtype="f8")  # average over iterations, now a (m, nparams) array
    # average value of all chains
    avg_phi = np.mean(chains, axis=(0, 1), dtype="f8")  # average over iterations and chains, now a (nparams,) array

    B = n / (m - 1.0) * np.sum((avg_phi_j - avg_phi) ** 2, axis=0, dtype="f8")  # now a (nparams,) array

    s2j = 1. / (n - 1.) * np.sum((chains - avg_phi_j) ** 2, axis=0, dtype="f8")  # now a (m, nparams) array

    W = 1. / m * np.sum(s2j, axis=0, dtype="f8")  # now a (nparams,) arary

    var_hat = (n - 1.) / n * W + B / n  # still a (nparams,) array
    std_hat = np.sqrt(var_hat)

    R_hat = np.sqrt(var_hat / W)  # still a (nparams,) array

    data = Table({"Value"      : avg_phi,
                  "Uncertainty": std_hat},
                 names=["Value", "Uncertainty"])

    print(data)

    ascii.write(data, sys.stdout, Writer=ascii.Latex, formats={"Value": "%0.3f", "Uncertainty": "%0.3f"})  #

    # print("Average parameter value: {}".format(avg_phi))
    # print("std_hat: {}".format(np.sqrt(var_hat)))
    print("R_hat: {}".format(R_hat))

    if np.any(R_hat >= 1.1):
        print("You might consider running the chain for longer. Not all R_hats are less than 1.1.")


def estimate_covariance(flatchain, base, ndim=0):
    if ndim == 0:
        d = flatchain.shape[1]
    else:
        d = ndim

    import matplotlib.pyplot as plt

    # print("Parameters {}".format(flatchain.param_tuple))
    # samples = flatchain.samples
    cov = np.cov(flatchain, rowvar=0)

    # Now try correlation coefficient
    cor = np.corrcoef(flatchain, rowvar=0)
    print("Correlation coefficient")
    print(cor)

    # Make a plot of correlation coefficient.

    fig, ax = plt.subplots(figsize=(0.5 * d, 0.5 * d), nrows=1, ncols=1)
    ext = (0.5, d + 0.5, 0.5, d + 0.5)
    ax.imshow(cor, origin="upper", vmin=-1, vmax=1, cmap="bwr", interpolation="none", extent=ext)
    fig.savefig("cor_coefficient.png")

    print("'Optimal' jumps with covariance (units squared)")

    opt_jump = 2.38 ** 2 / d * cov
    # opt_jump = 1.7**2/d * cov # gives about ??
    print(opt_jump)

    print("Standard deviation")
    std_dev = np.sqrt(np.diag(cov))
    print(std_dev)

    print("'Optimal' jumps")
    print(2.38 / np.sqrt(d) * std_dev)

    np.save(base + "opt_jump.npy", opt_jump)


def cat_list(file, flatchainList):
    '''
    Given a list of flatchains, concatenate all of these and write them to a
    single HDF5 file.
    '''
    # Write this out to the new file
    print("Opening", file)
    hdf5 = h5py.File(file, "w")

    cat = np.concatenate(flatchainList, axis=0)

    dset = hdf5.create_dataset("samples", cat.shape, compression='gzip',
                               compression_opts=9)
    dset[:] = cat
    # dset.attrs["parameters"] = "{}".format(param_tuple)

    hdf5.close()


log_lam_kws = frozenset(("CDELT1", "CRVAL1", "NAXIS1"))
flux_units = frozenset(("f_lam", "f_nu"))


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


def create_log_lam_grid(dv, wl_start=3000., wl_end=13000.):
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

    CDELT_temp = np.log10(dv / C.c_kms + 1.)
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


def create_mask(wl, fname):
    """
    Given a wavelength array (1D or 2D) and an ascii file containing the regions
    that one wishes to mask out, return a boolean array of indices for which
    wavelengths to KEEP in the calculation.

    :param wl: wavelength array (in AA)
    :param fname: filename of masking array

    :returns mask: boolean mask
    """
    data = ascii.read(fname)

    ind = np.ones_like(wl, dtype="bool")

    for row in data:
        # starting and ending indices
        start, end = row
        print(start, end)

        # All region of wavelength that do not fall in this range
        ind = ind & ((wl < start) | (wl > end))

    return ind
