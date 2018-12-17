# Given the astro seismic properties for Teff, logg, Z, go ahead and fit the spectra for vz, vsini, logOmega, and chebyshev coefficients.

# Because we will only be fitting a small number of parameters at once, I think we should go for a numerical optimizer.

# Write a simple optimizer to optimize the RV, Vsini, logOmega, and chebyshev parameters. Make sure that instead of transferring the model to the data rest frame, instead translate the RV to the rest frame air.

# We could also use emcee to check, later

import gc
import json
from itertools import chain

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import j1

import Starfish.constants as C
import Starfish.grid_tools
from Starfish import config
from Starfish.emulator import Emulator
from Starfish.spectrum import DataSpectrum, ChebyshevSpectrum

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
# ind = (wl > 5165.) & (wl < 5185.)
# wl = wl[ind]
#
fl = dataSpec.fls[0] #[ind]
sigma = dataSpec.sigmas[0] #[ind]
# mask = dataSpec.masks[0][ind]
ndata = len(wl)

print("ndata", ndata)
print("Data wl range", wl[0], wl[-1])

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

# For each star


# In the config file, list the astroseismic parameters as the starting grid parameters
# Read this into a ThetaParam object
grid = np.array(config["Theta"]["grid"])
# Now update the parameters for the emulator
# If pars are outside the grid, Emulator will raise C.ModelError
emulator.params = grid
mus, C_GP = emulator.matrix

npoly = config["cheb_degree"]
chebyshevSpectrum = ChebyshevSpectrum(dataSpec, 0, npoly=npoly)
chebyshevSpectrum.update(np.array(config["chebs"]))

def lnprob(p):
    vz, vsini, logOmega = p[:3]
    cheb = p[3:]

    chebyshevSpectrum.update(cheb)

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
        raise C.ModelError("vsini must be positive")
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

    # Get the mean spectrum
    X = (chebyshevSpectrum.k * flux_std * np.eye(ndata)).dot(eigenspectra.T)

    mean_spec = chebyshevSpectrum.k * flux_mean + X.dot(mus)
    R = fl - mean_spec

    # Evaluate chi2
    lnp = -0.5 * np.sum((R/sigma)**2)
    return [lnp, mean_spec, R]

def fprob(p):
    print(p)
    try:
        lnp = -lnprob(p)[0]
        return lnp
    except C.ModelError as e:
        return 1e99


def optimize():
    start = config["Theta"]
    p0 = np.concatenate((np.array([start["vz"], start["vsini"], start["logOmega"]]), np.zeros(npoly-1)))
    print("p0", p0)

    from scipy.optimize import fmin
    p = fmin(fprob, p0, maxiter=10000, maxfun=10000)
    print(p)

def generate():
    start = config["Theta"]
    p0 = np.concatenate((np.array([start["vz"], start["vsini"], start["logOmega"]]), np.zeros(npoly-1)))

    lnp, mean_spec, R = lnprob(p0)

    # Using RV, shift wl to zero velocity (opposite sign from before)
    wl_shift = wl * np.sqrt((C.c_kms - start["vz"]) / (C.c_kms + start["vz"]))

    # Write these to JSON
    my_dict = {"wl":wl_shift.tolist(), "data":fl.tolist(), "model":mean_spec.tolist(), "resid":R.tolist(), "sigma":sigma.tolist(), "spectrum_id":0, "order":order}

    fname = config.specfmt.format(0, order)
    f = open(config.name + fname + "spec.json", 'w')
    json.dump(my_dict, f, indent=2, sort_keys=True)
    f.close()



# Later on, use the value of RV to shift the residuals back to restframe, and if necessary do some interpolation to resample it.
