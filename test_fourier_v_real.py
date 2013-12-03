import matplotlib.pyplot as plt
import numpy as np
import model as m
import PHOENIX_tools as pt
from astropy.io import ascii

__author__ = 'ian'

'''This package is designed to test the Fourier Techniques against the previously tested techniques to see the
difference, if any.'''

#Default parameters: T = 5900 K, logg = 3.5 dex, Fe/H = 0.0 dex

#Load raw spectrum
raw_wl = pt.w_full
raw_fl = pt.load_flux_full(6400., 4.5, "-0.5", norm=True)

#truncate to 5000 - 5400 Ang
raw_ind = (raw_wl > 4900) & (raw_wl < 5500)
#raw_ind = (raw_wl > 5180) & (raw_wl < 5200)
wl = raw_wl[raw_ind]
fl = raw_fl[raw_ind]

#ascii.write([wl, fl], 'raw.txt', names=['wl', 'fl'], Writer=ascii.NoHeader)

# Uncomment to show raw spectrum
#plt.plot(wl,fl)
#plt.show()

# Convolve to TRES
dlam = 0.01 # spacing of model points for TRES resolution kernel

#convolve with filter to resolution of TRES
filt = m.gauss_series(dlam, lam0=5150, V=6.8)
f_TRES = m.convolve(fl, filt)

wlm = np.load("wave_grid_2kms.npy")
#wlm = np.load('wave_grid_0.35kms.npy')

# Try doing Fourier method on 0.5 km/s sampled points.
flm = pt.resample_and_convolve(raw_fl[pt.ind],pt.wave_grid_fine, wlm)

#ind = (5100 < wlm) & (wlm < 5200)
#flm = flm[ind]
#wlm = wlm[ind]

#Try getting spectrum directly from the LIB
fl_LIB = m.flux(6390, 4.5, -0.49999)
wl_LIB = m.wave_grid
ind = (5100 < wl_LIB) & (wl_LIB < 5200)
fl_LIB = fl_LIB[ind]
wl_LIB = wl_LIB[ind]

#downsample real hi-res spectrum to wave-grid
f_interpolator = m.InterpolatedUnivariateSpline(wl, f_TRES)
f_real = f_interpolator(wl_LIB) #do spline interpolation to TRES pixels


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
ax[0].plot(wl_LIB, fl_LIB, "b", label="Fourier")
ax[0].plot(wl_LIB, f_real, "g", label="Real")
ax[0].legend()
ax[1].plot(wl_LIB, (fl_LIB - f_real)/fl_LIB)
plt.show()

#write spectra to ascii file to test FWHM w/ IRAF

#ascii.write([wlm, flm], 'fourier.txt', names=['wlm', 'flm'], Writer=ascii.FixedWidthNoHeader, delimiter=None)
#ascii.write([wlm, f_real], 'real.txt', names=['wlm', 'f_real'], Writer=ascii.NoHeader)



# Convolve to stellar broadening
#k = m.vsini_ang(5200, 5.0) # stellar rotation kernel centered at 5200, 5km/s vsini
#f_sb = m.convolve(f_TRES, k)

#TRES broaden 6.8 km/s

#With and without Stellar broadening
#Downsample



