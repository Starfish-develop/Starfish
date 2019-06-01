import logging

import numpy as np
from nptyping import Array
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

import Starfish.constants as C
from .kernels import global_covariance_matrix


def find_residual_peaks(
    model, num_residuals=100, threshold=4.0, buffer=2, wl_range=(0, np.inf)
):
    """
    Find the peaks of the most recent residual and return their properties to aid in setting up local kernels

    Parameters
    ----------
    model : Model 
        The model to determine peaks from. Need only have a residuals array.
    num_residuals : int, optional
        The number of residuals to average together for determining peaks. By default 100.
    threshold : float, optional
        The sigma clipping threshold, by default 4.0
    buffer : float, optional
        The minimum distance between peaks, in Angstrom, by default 2.0
    wl_range : 2-tuple
        The (min, max) wavelengths to consider. Default is (0, np.inf)

    Returns
    -------
    means : list
        The means of the found peaks, with the same units as model.data.waves
    """
    residual = np.mean(list(model.residuals)[-num_residuals:], axis=0)
    mask = np.abs(residual - residual.mean()) > threshold * residual.std()
    mask &= (model.data.waves > wl_range[0]) & (model.data.waves < wl_range[1])
    peak_waves = model.data.waves[1:-1][mask[1:-1]]
    mus = []
    covered = np.zeros_like(peak_waves, dtype=bool)
    # Sort from largest residual to smallest
    abs_resid = np.abs(residual[1:-1][mask[1:-1]])
    for wl, resid in sorted(
        zip(peak_waves, abs_resid), key=lambda t: t[1], reverse=True
    ):
        if wl in peak_waves[covered]:
            continue
        mus.append(wl)
        ind = (peak_waves >= (wl - buffer)) & (peak_waves <= (wl + buffer))
        covered |= ind

    return mus


def optimize_residual_peaks(model, mus, sigma0=50, num_residuals=100):
    """
    Optimize the local covariance parameters based on fitting the residual input means as Gaussians around the residuals

    Parameters
    ----------
    model : Model
        The model to determine peaks from. Need only have a residuals array.
    mus : array-like
        The means to instantiate Gaussians at and optimize.
    sigma0 : float, optional
        The initial standard deviation (in Angstrom) of each Gaussian. Default is 50 Angstrom.
    num_residuals : int, optional
            The number of residuals to average together for determining peaks. By default 100.

    Returns
    -------
    dict
        A dictionary of optimized parameters ready to be plugged into model.local
    """
    residual = np.mean(list(model.residuals)[-num_residuals:], axis=0)

    def chi2(P, wave, resid, sigma):
        log_amp, mu, log_sigma = P
        rr = C.c_kms / mu * np.abs(wave - mu)
        gauss = np.exp(log_amp - 0.5 * (rr / np.exp(log_sigma)) ** 2)
        R = gauss - resid
        return np.sum((R / sigma) ** 2)

    params = []

    for mu in mus:
        mask = (model.data.waves > mu - sigma0) & (model.data.waves < mu + sigma0)
        wave = model.data.waves[mask]
        resid = residual[mask]
        sigma = model.data.sigmas[mask]

        P0 = np.array([np.log(np.abs(resid).max()), mu, np.log(sigma0)])

        soln = minimize(chi2, P0, args=(wave, resid, sigma))

        params.append({"log_amp": soln.x[0], "mu": soln.x[1], "log_sigma": soln.x[2]})

    return params


log = logging.getLogger(__name__)


def covariance_debugger(cov: Array[float]):
    """
    Special debugging information for the covariance matrix decomposition.
    """
    log.info(f"{'Covariance Debugger':-^60}".format())
    log.info("See https://github.com/iancze/Starfish/issues/26")
    log.info("Covariance matrix at a glance:")
    if cov.diagonal().min() < 0.0:
        log.warning("- Negative entries on the diagonal:")
        log.info("\t- Check uncertainty estimates: should all be positive")
    elif np.any(np.isnan(cov.diagonal())):
        log.warning("- Covariance matrix has a NaN value on the diagonal")
    else:
        if not np.allclose(cov, cov.T):
            log.warning("- The covariance matrix is highly asymmetric")

        # Still might have an asymmetric matrix below `allclose` threshold
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        n_neg = (eigenvalues < 0).sum()
        n_tot = len(eigenvalues)
        log.info(f"- There are {n_neg} negative eigenvalues out of {n_tot}.")

        def mark(val):
            return ">" if val < 0 else "."

        log.info("Covariance matrix eigenvalues:")
        for i in range(10):
            log.info(
                "{: >6} {:{fill}>20.3e}".format(
                    i, eigenvalues[i], fill=mark(eigenvalues[i])
                )
            )
        log.info("{: >15}".format("..."))
        for i in range(10):
            log.info(
                "{: >6} {:{fill}>20.3e}".format(
                    n_tot - 10 + i,
                    eigenvalues[-10 + i],
                    fill=mark(eigenvalues[-10 + i]),
                )
            )

    log.info(f"{'-':-^60}")
