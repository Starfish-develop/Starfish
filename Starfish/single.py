# Serial implementation for sampling a single chunk of a spectrum, namely one or two spectroscopic lines.

import multiprocessing as mp

import argparse
parser = argparse.ArgumentParser(prog="single.py", description="Run Starfish fitting model in parallel.")
parser.add_argument("-r", "--run_index", help="All data will be written into this directory, overwriting any that exists. Default is current working directory.")
# Even though these arguments aren't being used, we need to add them.
parser.add_argument("--generate", action="store_true", help="Write out the data, mean model, and residuals for current parameter settings.")
# parser.add_argument("--initPhi", action="store_true", help="Create *phi.json files for each order using values in config.yaml")
parser.add_argument("--optimize", action="store_true", help="Optimize the parameters.")
parser.add_argument("--sample", action="store_true", help="Sample the parameters.")
parser.add_argument("--samples", type=int, default=5, help="How many samples to run?")
parser.add_argument("--cpus", type=int, default=mp.cpu_count(), help="How many threads to use for emcee sampling.")
args = parser.parse_args()

import os
import numpy as np

from Starfish import config
import Starfish.grid_tools
from Starfish.samplers import StateSampler
from Starfish.spectrum import DataSpectrum, Mask, ChebyshevSpectrum
from Starfish.utils import create_mask
from Starfish.emulator import Emulator
import Starfish.constants as C
from Starfish.covariance import get_dense_C, make_k_func, make_k_func_region
from Starfish.model import ThetaParam, PhiParam

from scipy.special import j1
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet

import gc
import logging

from itertools import chain
from collections import deque
from operator import itemgetter
import yaml
import shutil
import json

def init_directories(run_index=None):
    '''
    If we are sampling, then we need to setup output directories to store the
    samples and other output products. If we're sampling, we probably want to be
    running multiple chains at once, and so we have to set things up so that they
    don't conflict.

    :returns: routdir, the outdir for this current run.
    '''

    base = config.outdir + config.name + "/run{:0>2}/"

    if run_index is None:
        run_index = 0
        while os.path.exists(base.format(run_index)):
            print(base.format(run_index), "exists")
            run_index += 1
        routdir = base.format(run_index)

    else:
        routdir = base.format(run_index)
        #Delete this routdir, if it exists
        if os.path.exists(routdir):
            print("Deleting", routdir)
            shutil.rmtree(routdir)

    print("Creating ", routdir)
    os.makedirs(routdir)

    # Copy yaml file to routdir for archiving purposes
    shutil.copy("config.yaml", routdir + "config.yaml")


    return routdir

if args.run_index:
    config.routdir = init_directories(args.run_index)
else:
    config.routdir = ""

# list of keys from 0 to (norders - 1)
# For now, we will only load one order because we plan on fitting very narrow chunks of the spectrum

orders = config.data["orders"]
assert len(orders) == 1, "Can only use 1 order for now."
order = orders[0]

# Load just this order for now.
dataSpec = DataSpectrum.open(config.data["files"][0], orders=config.data["orders"])
instrument = eval("Starfish.grid_tools." + config.data["instruments"][0])()

# full_mask = create_mask(dataSpec.wls, config.data["masks"][0])
# dataSpec.add_mask(full_mask)

wl = dataSpec.wls[0]

# Truncate these to our shorter range to make it faster
ind = (wl > 5165.) & (wl < 5185.)
wl = wl #[ind]

fl = dataSpec.fls[0] #[ind]
sigma = dataSpec.sigmas[0] #[ind]
# mask = dataSpec.masks[0] #[ind]
mask = np.ones_like(wl, dtype="bool")
ndata = len(wl)

print("ndata", ndata)

# Set up the emulator for this chunk
emulator = Emulator.open()
emulator.determine_chunk_log(wl)

pca = emulator.pca

wl_FFT_orig = pca.wl

print("FFT length", len(wl_FFT_orig))
print(wl_FFT_orig[0], wl_FFT_orig[-1])

# The raw eigenspectra and mean flux components
EIGENSPECTRA = np.vstack((pca.flux_mean[np.newaxis,:], pca.flux_std[np.newaxis,:], pca.eigenspectra))

ss = np.fft.rfftfreq(pca.npix, d=emulator.dv)
ss[0] = 0.01 # junk so we don't get a divide by zero error

sigma_mat = sigma**2 * np.eye(ndata)
mus, C_GP, data_mat = None, None, None


# This single line will be sampled by emcee
def lnprob(p):

    grid = p[:3]
    vz, vsini, logOmega = p[3:]

    # Local, shifted copy of wavelengths
    wl_FFT = wl_FFT_orig * np.sqrt((C.c_kms + vz) / (C.c_kms - vz))

    # Holders to store the convolved and resampled eigenspectra
    eigenspectra = np.empty((pca.m, ndata))
    flux_mean = np.empty((ndata,))
    flux_std = np.empty((ndata,))

    # If vsini is less than 0.2 km/s, we might run into issues with
    # the grid spacing. Therefore skip the convolution step if we have
    # values smaller than this.
    # FFT and convolve operations
    if vsini < 0.0:
        return -np.inf
    elif vsini < 0.2:
        # Skip the vsini taper due to instrumental effects
        eigenspectra_full = EIGENSPECTRA.copy()
    else:
        FF = np.fft.rfft(EIGENSPECTRA, axis=1)

        # Determine the stellar broadening kernel
        ub = 2. * np.pi * vsini * ss
        sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
        # set zeroth frequency to 1 separately (DC term)
        sb[0] = 1.

        # institute vsini taper
        FF_tap = FF * sb

        # do ifft
        eigenspectra_full = np.fft.irfft(FF_tap, pca.npix, axis=1)

    # Spectrum resample operations
    if min(wl) < min(wl_FFT) or max(wl) > max(wl_FFT):
        raise RuntimeError("Data wl grid ({:.2f},{:.2f}) must fit within the range of wl_FFT ({:.2f},{:.2f})".format(min(wl), max(wl), min(wl_FFT), max(wl_FFT)))

    # Take the output from the FFT operation (eigenspectra_full), and stuff them
    # into respective data products
    for lres, hres in zip(chain([flux_mean, flux_std], eigenspectra), eigenspectra_full):
        interp = InterpolatedUnivariateSpline(wl_FFT, hres, k=5)
        lres[:] = interp(wl)
        del interp

    gc.collect()

    # Adjust flux_mean and flux_std by Omega
    Omega = 10**logOmega
    flux_mean *= Omega
    flux_std *= Omega

    # Now update the parameters from the emulator
    # If pars are outside the grid, Emulator will raise C.ModelError
    try:
        emulator.params = grid
        mus, C_GP = emulator.matrix
    except C.ModelError:
        return -np.inf

    # Get the mean spectrum
    X = (flux_std * np.eye(ndata)).dot(eigenspectra.T)
    mean_spec = flux_mean + X.dot(mus)
    R = fl - mean_spec

    # Evaluate chi2
    lnp = -0.5 * np.sum((R[mask]/sigma[mask])**2)
    return lnp

    # Necessary to interpolate over all spectru
    #
    # CC = X.dot(C_GP.dot(X.T)) + data_mat
    #
    # try:
    #
    #     factor, flag = cho_factor(CC)
    #
    #     R = self.fl - self.chebyshevSpectrum.k * self.flux_mean - X.dot(self.mus)
    #
    #     logdet = np.sum(2 * np.log((np.diag(factor))))
    #     self.lnprob = -0.5 * (np.dot(R, cho_solve((factor, flag), R)) + logdet)
    #
    #     self.logger.debug("Evaluating lnprob={}".format(self.lnprob))
    #     return self.lnprob
    #
    # # To give us some debugging information about what went wrong.
    # except np.linalg.linalg.LinAlgError:
    #     print("Spectrum:", self.spectrum_id, "Order:", self.order)
    #     raise
