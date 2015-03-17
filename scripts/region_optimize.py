#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(prog="region_optimize.py", description="Find the kernel parameters for Gaussian region zones.")
parser.add_argument("spectrum", help="JSON file containing the data, model, and residual.")
parser.add_argument("regions", help="JSON file containing the mu of each spectral line.")
parser.add_argument("--sigma0", type=float, default=2, help="Range over which regions can't overlap.")
args = parser.parse_args()

import json
import numpy as np
from scipy.optimize import fmin
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet

import Starfish
from Starfish.model import PhiParam
from Starfish.covariance import get_dense_C, make_k_func

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

f = open(args.regions, "r")
read = json.load(f) # read is a dictionary
f.close()

mus = np.array(read["mus"])
assert spectrum_id == read["spectrum_id"], "Spectrum/Order mismatch"
assert order == read["order"], "Spectrum/Order mismatch"

# Load the guesses for the global parameters from the .json
# If the file exists, optionally initiliaze to the chebyshev values
fname = Starfish.specfmt.format(spectrum_id, order) + "phi.json"

try:
    phi = PhiParam.load(fname)
except FileNotFoundError:
    print("No order parameter file found (e.g. sX_oXX_phi.json), please run an optimizer first.")
    raise

def optimize_region(wl, residuals, sigma, mu):

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


    par = Starfish.config["region_params"]
    p0 = np.array([par["logAmp"], par["sigma"]])

    f = lambda x: -fprob(x)

    try:
        p = fmin(f, p0, maxiter=10000, maxfun=10000)
        # print(p)
        return p
    except np.linalg.linalg.LinAlgError:
        return p0

# optimize_region(wl, resid, sigma, mus[4])
for mu in mus:
    print(mu, optimize_region(wl, resid, sigma, mu))
