#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(prog="region_optimize.py", description="Find the kernel parameters for Gaussian region zones.")
parser.add_argument("spectrum", help="JSON file containing the data, model, and residual.")
parser.add_argument("--sigma0", type=float, default=2, help="(AA) to use in fitting")
args = parser.parse_args()

import json
import numpy as np
from scipy.optimize import fmin
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet

from Starfish import config
from Starfish.model import PhiParam
from Starfish.covariance import get_dense_C, make_k_func
from Starfish import constants as C

# Load the spectrum and then take the data products.
f = open(args.spectrum, "r")
read = json.load(f) # read is a dictionary
f.close()

wl = np.array(read["wl"])
# data_full = np.array(read["data"])
# model = np.array(read["model"])
resid = np.array(read["resid"])
sigma = np.array(read["sigma"])
spectrum_id = read["spectrum_id"]
order = read["order"]

fname = config.specfmt.format(spectrum_id, order) + "regions.json"
f = open(fname, "r")
read = json.load(f) # read is a dictionary
f.close()

mus = np.array(read["mus"])
assert spectrum_id == read["spectrum_id"], "Spectrum/Order mismatch"
assert order == read["order"], "Spectrum/Order mismatch"

# Load the guesses for the global parameters from the .json
# If the file exists, optionally initiliaze to the chebyshev values
fname = config.specfmt.format(spectrum_id, order) + "phi.json"

try:
    phi = PhiParam.load(fname)
except FileNotFoundError:
    print("No order parameter file found (e.g. sX_oXXphi.json), please run `star.py --initPhi` first.")
    raise

# Puposely set phi.regions to none for this exercise, since we don't care about existing regions, and likely we want to overwrite them.
phi.regions = None

def optimize_region_residual(wl, residuals, sigma, mu):
    '''
    Determine the optimal parameters for the line kernels by fitting a Gaussian directly to the residuals.
    '''
    # Using sigma0, truncate the wavelength vector and residulas to include
    # only those portions that fall in the range [mu - sigma, mu + sigma]
    ind = (wl > mu - args.sigma0) & (wl < mu + args.sigma0)
    wl = wl[ind]
    R = residuals[ind]
    sigma = sigma[ind]

    sigma_mat = phi.sigAmp * sigma**2 * np.eye(len(wl))

    max_r = 6.0 * phi.l # [km/s]
    k_func = make_k_func(phi)

    # Use the full covariance matrix when doing the likelihood eval
    CC = get_dense_C(wl, k_func=k_func, max_r=max_r) + sigma_mat
    factor, flag = cho_factor(CC)
    logdet = np.sum(2 * np.log((np.diag(factor))))

    rr = C.c_kms/mu * np.abs(mu - wl) # Km/s

    def fprob(p):
        # The likelihood function

        # Requires sign about amplitude, so we can't use log.
        amp, sig = p

        gauss = amp * np.exp(-0.5 * rr**2/sig**2)

        r = R - gauss

        # Create a Gaussian using these parameters, and re-evaluate the residual
        lnprob = -0.5 * (np.dot(r, cho_solve((factor, flag), r)) + logdet)
        return lnprob

    par = config["region_params"]
    p0 = np.array([10**par["logAmp"], par["sigma"]])

    f = lambda x: -fprob(x)

    try:
        p = fmin(f, p0, maxiter=10000, maxfun=10000, disp=False)
        # print(p)
        return p
    except np.linalg.linalg.LinAlgError:
        return p0



def optimize_region_covariance(wl, residuals, sigma, mu):
    '''
    Determine the optimal parameters for the line kernels by actually using a chunk of the covariance matrix.

    Note this actually uses the assumed global parameters.
    '''

    # Using sigma0, truncate the wavelength vector and residulas to include
    # only those portions that fall in the range [mu - sigma, mu + sigma]
    ind = (wl > mu - args.sigma0) & (wl < mu + args.sigma0)
    wl = wl[ind]
    R = residuals[ind]
    sigma = sigma[ind]

    sigma_mat = phi.sigAmp * sigma**2 * np.eye(len(wl))

    max_rl = 6.0 * phi.l # [km/s]

    # Define a probability function for the residuals
    def fprob(p):
        logAmp, sigma = p

        # set phi.regions = p
        phi.regions = np.array([logAmp, mu, sigma])[np.newaxis, :]

        max_rr = 4.0 * sigma
        max_r = max(max_rl, max_rr)

        k_func = make_k_func(phi)

        CC = get_dense_C(wl, k_func=k_func, max_r=max_r) + sigma_mat

        factor, flag = cho_factor(CC)
        logdet = np.sum(2 * np.log((np.diag(factor))))
        lnprob = -0.5 * (np.dot(R, cho_solve((factor, flag), R)) + logdet)

        # print(p, lnprob)
        return lnprob


    par = config["region_params"]
    p0 = np.array([par["logAmp"], par["sigma"]])

    f = lambda x: -fprob(x)

    try:
        p = fmin(f, p0, maxiter=10000, maxfun=10000)
        print(p)
        return p
    except np.linalg.linalg.LinAlgError:
        return p0

# Regions will be a 2D array with shape (nregions, 3)
regions = []
for mu in mus:
    # amp, sig = optimize_region_residual(wl, resid, sigma, mu)
    # regions.append([np.log10(np.abs(amp)), mu, sig])

    logAmp, sig = optimize_region_covariance(wl, resid, sigma, mu)
    regions.append([logAmp, mu, sig])

# Add these values back to the phi parameter file and save
phi.regions = np.array(regions)
phi.save()
