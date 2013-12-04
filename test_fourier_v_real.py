import matplotlib.pyplot as plt
import numpy as np
import model as m
import PHOENIX_tools as pt
from astropy.io import ascii
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
import h5py
from scipy.spatial import Delaunay
from scipy.ndimage.interpolation import map_coordinates


__author__ = 'ian'

'''This package is designed to test the Fourier Techniques against the previously tested techniques to see the
difference, if any.'''

#Default parameters: T = 5900 K, logg = 3.5 dex, Fe/H = 0.0 dex

#Load raw spectrum
#raw_wl = pt.w_full
#raw_fl = pt.load_flux_full(6400., 4.0, "-0.5", norm=True)
#raw_fl2 = pt.load_flux_full(6300., 4.0, "-0.5", norm=True)

#truncate to 5000 - 5400 Ang
#raw_ind = (raw_wl > 4900) & (raw_wl < 5500)
#raw_ind = (raw_wl > 5180) & (raw_wl < 5200)
#wl = raw_wl[raw_ind]
#fl = raw_fl[raw_ind]

#fl2 = raw_fl2[raw_ind]

#plt.plot(wl,fl)
#plt.plot(wl,fl2)
#plt.show()

#ascii.write([wl, fl], 'raw.txt', names=['wl', 'fl'], Writer=ascii.NoHeader)

# Uncomment to show raw spectrum
#plt.plot(wl,fl)
#plt.show()

# Convolve to TRES
#dlam = 0.01 # spacing of model points for TRES resolution kernel

#convolve with filter to resolution of TRES
#filt = m.gauss_series(dlam, lam0=5150, V=6.8)
#f_TRES = m.convolve(fl, filt)
#
wl = np.load("wave_grid_2kms.npy")
#wlm = np.load('wave_grid_0.35kms.npy')

#wl_LIB = m.wave_grid
#fl_LIB = m.flux(6400.0001, 4.0, -0.5)

#Create own LinearNDInterpolator
#
#f_6300_40_n05 = pt.process_spectrum([6300, 4.0, "-0.5"])[m.ind]
#f_6300_40_0 = pt.process_spectrum([6300, 4.0, "-0.0"])[m.ind]
#f_6300_45_n05 = pt.process_spectrum([6300, 4.5, "-0.5"])[m.ind]
#f_6300_45_0 = pt.process_spectrum([6300, 4.5, "-0.0"])[m.ind]
#
#f_6400_40_n05 = pt.process_spectrum([6400, 4.0, "-0.5"])[m.ind]
#f_6400_40_0 = pt.process_spectrum([6400, 4.0, "-0.0"])[m.ind]
#f_6400_45_n05 = pt.process_spectrum([6400, 4.5, "-0.5"])[m.ind]
#f_6400_45_0 = pt.process_spectrum([6400, 4.5, "-0.0"])[m.ind]
#
#f_6500_40_n05 = pt.process_spectrum([6500, 4.0, "-0.5"])[m.ind]
#f_6500_40_0 = pt.process_spectrum([6500, 4.0, "-0.0"])[m.ind]
#f_6500_45_n05 = pt.process_spectrum([6500, 4.5, "-0.5"])[m.ind]
#f_6500_45_0 = pt.process_spectrum([6500, 4.5, "-0.0"])[m.ind]
#
#fluxes = np.array([f_6300_40_n05, f_6300_40_0, f_6300_45_n05, f_6300_45_0, f_6400_40_n05, f_6400_40_0,
#f_6400_45_n05, f_6400_45_0, f_6500_40_n05, f_6500_40_0, f_6500_45_n05, f_6500_45_0])
##print(len(fluxes), fluxes.shape)
#
#points = np.array([[6300, 4.0, -0.5], [6300, 4.0, 0.0], [6300, 4.5, -0.5], [6300, 4.5, 0.0],
#                   [6400, 4.0, -0.5], [6400, 4.0, 0.0], [6400, 4.5, -0.5], [6400, 4.5, 0.0],
#                   [6500, 4.0, -0.5], [6500, 4.0, 0.0], [6500, 4.5, -0.5], [6500, 4.5, 0.0]])
#
#xpoints = np.array([6300, 6400, 6500])
#ypoints = np.array([4.0, 4.5])
#
#intp = RectBivariateSpline(xpoints, ypoints, np.array([[f_6300_40_n05, f_6300_45_n05],
#                                                       [f_6400_40_n05, f_6400_45_n05], [f_6500_40_n05, f_6500_45_n05]]))
#flux = intp(6350, 4.25)
#print(flux)

#flux = LinearNDInterpolator(points, fluxes, fill_value=1.)

# Try doing Fourier method on 0.5 km/s sampled points.
#flm = pt.resample_and_convolve(raw_fl[pt.ind],pt.wave_grid_fine, wlm)
#
#plt.plot(wlm, (flm - fl_LIB)/flm)
#plt.show()

#ind = (5100 < wlm) & (wlm < 5200)
#flm = flm[ind]
#wlm = wlm[ind]

#Creating a flux interpolator from spectra read directly from LIB
#Designed to match the same as points
#pointers = [[39, 8, 0], [39, 8, 1], [39, 9, 0], [39, 9, 1],
#            [40, 8, 0], [40, 8, 1], [40, 9, 0], [40, 9, 1],
#            [41, 8, 0], [41, 8, 1], [41, 9, 0], [41, 9, 1]]
#
#fhdf5 = h5py.File('LIB_2kms.hdf5', 'r')
#LIB = fhdf5['LIB']
#print(map_coordinates(LIB, np.array([40, 8.5, 0.5])))
#
#fluxes = np.empty((12, np.sum(m.ind)))
#for i in range(12):
#    t, l, z = pointers[i]
#
#    fluxes[i] = LIB[t, l, z][m.ind]
#
#flux = LinearNDInterpolator(points, fluxes, fill_value=1.)

#Temp = 6400
#deltaT = 1e-3
#wl_LIB = m.wave_grid
#fl_LIB = flux(Temp + deltaT, 4.25, -0.25)
#fl_LIB2 = flux(Temp - deltaT, 4.25, -0.25)
#print("Tupper :", Temp + deltaT)
#print("Tlower :", Temp - deltaT)
# Using the very same files from the LIB gives ~2e-6 amplitude and 4e-3 total deviation over 5-6000 A, the same as
# loading raw.


#Using model.flux_interpolator()
Temp = 6400
deltaT = 0
logg = 4.0
delta_L = 1e-4
Z = 0.
deltaZ = 0

wl_LIB = m.wave_grid
fl_LIB = m.flux(Temp + deltaT, logg + delta_L, Z + deltaZ)
fl_LIB2 = m.flux(Temp - deltaT, logg - delta_L, Z - deltaZ)
print("Tupper :", Temp + deltaT)
print("Tlower :", Temp - deltaT)
# Above gives ~0.02 amplitude and 17 total deviation over 5-6000 A


#Using flux interpolator among 12 points loaded in this module
#Temp = 6400
#deltaT = 1e-3
#wl_LIB = m.wave_grid
#fl_LIB = flux(Temp + deltaT, 4.25, -0.25)
#fl_LIB2 = flux(Temp - deltaT, 4.25, -0.25)
#print("Tupper :", Temp + deltaT)
#print("Tlower :", Temp - deltaT)
# Above gives ~2e-6 amplitude and 4e-3 total deviation over 5-6000 A


plt.plot(wl_LIB, (fl_LIB - fl_LIB2)/fl_LIB)
plt.title("Total deviation: %s" % np.sum((fl_LIB - fl_LIB2)/fl_LIB))
#plt.plot(wl_LIB, fl_LIB2)
#plt.plot(wl_LIB, fl_LIB3)
plt.show()

#ind = (5100 < wl_LIB) & (wl_LIB < 5200)

#fl_LIB = fl_LIB[ind]
#wl_LIB = wl_LIB[ind]

#downsample real hi-res spectrum to wave-grid
#f_interpolator = m.InterpolatedUnivariateSpline(wl, f_TRES)
#f_real = f_interpolator(wl_LIB) #do spline interpolation to TRES pixels


#fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
#ax[0].plot(wl_LIB, fl_LIB, "b", label="Fourier")
#ax[0].plot(wl_LIB, f_real, "g", label="Real")
#ax[0].legend()
#ax[1].plot(wl_LIB, (fl_LIB - f_real)/fl_LIB)
#plt.show()



#write spectra to ascii file to test FWHM w/ IRAF

#ascii.write([wlm, flm], 'fourier.txt', names=['wlm', 'flm'], Writer=ascii.FixedWidthNoHeader, delimiter=None)
#ascii.write([wlm, f_real], 'real.txt', names=['wlm', 'f_real'], Writer=ascii.NoHeader)



# Convolve to stellar broadening
#k = m.vsini_ang(5200, 5.0) # stellar rotation kernel centered at 5200, 5km/s vsini
#f_sb = m.convolve(f_TRES, k)

#TRES broaden 6.8 km/s

#With and without Stellar broadening
#Downsample



