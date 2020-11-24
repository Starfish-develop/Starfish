import logging

import numpy as np
from nptyping import NDArray
from scipy.optimize import minimize
import scipy.stats as st

import Starfish.constants as C
from .kernels import global_covariance_matrix


def find_residual_peaks(
    model, num_residuals=100, threshold=4.0, buffer=2, wl_range=(0, np.inf)
):
    """
    Find the peaks of the most recent residual and return their properties to aid in
    setting up local kernels

    Parameters
    ----------
    model : Model
        The model to determine peaks from. Need only have a residuals array.
    num_residuals : int, optional
        The number of residuals to average together for determining peaks. By default
        100.
    threshold : float, optional
        The sigma clipping threshold, by default 4.0
    buffer : float, optional
        The minimum distance between peaks, in Angstrom, by default 2.0
    wl_range : 2-tuple
        The (min, max) wavelengths to consider. Default is (0, np.inf)

    Returns
    -------
    means : list
        The means of the found peaks, with the same units as model.data.wave
    """
    residual = np.mean(list(model.residuals)[-num_residuals:], axis=0)
    sigma = residual.std()
    if "global_cov" in model.params:
        ag = np.exp(model.params["global_cov:log_amp"])
        lg = np.exp(model.params["global_cov:log_ls"])
        sigma += global_covariance_matrix(
            model.data.wave, model["T"], ag, lg
        ).diagonal()
    mask = np.abs(residual - residual.mean()) > threshold * sigma
    mask &= (model.data.wave > wl_range[0]) & (model.data.wave < wl_range[1])
    peak_waves = model.data.wave[1:-1][mask[1:-1]]
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


def optimize_residual_peaks(model, mus, threshold=0.1, sigma0=50, num_residuals=100):
    """
    Optimize the local covariance parameters based on fitting the residual input means
    as Gaussians around the residuals

    Parameters
    ----------
    model : Model
        The model to determine peaks from. Need only have a residuals array.
    mus : array-like
        The means to instantiate Gaussians at and optimize.
    threshold : float, optional
        This is the threshold for restricting kernels; i.e. if a fit amplitude is less
        than threshold standard deviations then it will be thrown away. Default is 0.1
    sigma0 : float, optional
        The initial standard deviation (in Angstrom) of each Gaussian. Default is 50
        Angstrom.
    num_residuals : int, optional
        The number of residuals to average together for determining peaks. By default
        100.

    Returns
    -------
    dict
        A dictionary of optimized parameters ready to be plugged into model["local_cov"]

    Warning
    -------
    I have had inconsistent results with this optimization, be mindful of your outputs
    and consider hand-tuning after optimizing.
    """
    residual = np.mean(list(model.residuals)[-num_residuals:], axis=0)
    amp_cutoff = threshold * residual.std()
    if "global_cov" in model.params:
        ag = np.exp(model.params["global_cov:log_amp"])
        lg = np.exp(model.params["global_cov:log_ls"])
        global_cov = global_covariance_matrix(model.data.wave, model["T"], ag, lg)
    else:
        global_cov = None

    def chi2(P, wave, resid, sigma):
        log_amp, mu, log_sigma = P
        _amp = np.exp(log_amp)
        _sigma = np.exp(log_sigma)
        if _amp == 0 or _sigma == 0:
            return np.Inf
        # Logistic prior for the widths out to 2sigma
        prior = st.uniform.logpdf(_sigma, 0, 2 * sigma0)
        # Put prior on widths and heights such that the integrated area should be less
        # than the trapezoidal area of the residual
        area_max = 2 * _sigma * np.abs(resid).max()
        # Area under a gaussian
        # https://en.wikipedia.org/wiki/Gaussian_function#Integral_of_a_Gaussian_function
        area = np.sqrt(2 * np.pi * _sigma) * _amp
        prior += st.uniform.logpdf(area, 0, area_max)
        prior += st.uniform.logpdf(_amp, 0, np.abs(resid).max())
        rr = C.c_kms / mu * np.abs(wave - mu)
        gauss = _amp * np.exp(-0.5 * (rr / _sigma) ** 2)
        R = gauss - resid
        return np.sum((R / sigma) ** 2) - prior

    params = []

    for mu in mus:
        mask = (model.data.wave > mu - sigma0) & (model.data.wave < mu + sigma0)
        wave = model.data.wave[mask]
        resid = residual[mask]
        sigma = model.data.sigma[mask]
        if global_cov is not None:
            sigma += global_cov.diagonal()[mask]

        P0 = np.array([np.log(np.abs(resid).max()), mu, np.log(sigma0)])

        soln = minimize(
            chi2,
            P0,
            args=(wave, resid, sigma),
            method="Nelder-Mead",
            options=dict(maxiter=1000),
        )
        if soln.x[0] > np.log(amp_cutoff):
            params.append(
                {"log_amp": soln.x[0], "mu": soln.x[1], "log_sigma": soln.x[2]}
            )

    return sorted(params, key=lambda s: s["mu"])


log = logging.getLogger(__name__)


def covariance_debugger(cov: NDArray[float]):
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
