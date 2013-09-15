import numpy as np
import matplotlib.pyplot as plt
from PHOENIX_tools import load_flux_full,w_full
from deredden import deredden
from astropy.io import fits
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from PHOENIX_tools import load_flux_full,w_full

filt = fits.open("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/landolt_v_004_syn.fits")[1].data
#filt = fits.open("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/landolt_i_004_syn.fits")[1].data
wl = filt['WAVELENGTH']
trans = filt['THROUGHPUT']
err = filt['ERROR']

def plot_filter():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wl,trans)
    plt.show()

#plot_filter()

ind = (w_full > 2000) & (w_full < 40000)
ww = w_full[ind]

ff = load_flux_full(5900,3.5)[ind]*2e-28
#redden spectrum
#red = ff/deredden(ww,1.5,mags=False)

#interpolate filter response to model spacing
filt_interp = interp1d(wl,trans,kind='cubic')
filt_ind = (ww > wl[0]) & (ww < wl[-1])
wl_filt = ww[filt_ind]
fl_filt = ff[filt_ind]
S = filt_interp(wl_filt)
Si = wl_filt * S
const = trapz(Si,wl_filt)
print(const)
S_cn = Si/const

'''Si may be obfuscating.'''

#f_lam = trapz(fl_filt * S, wl_filt)/trapz(S,wl_filt)
#f_lam2 = trapz(fl_filt * S * wl_filt, wl_filt)/trapz(S * wl_filt, wl_filt)
#print(f_lam)
#print(f_lam2)

def calc_flux():
    return trapz(fl_filt * S_cn, wl_filt)

def main():
    for i in range(5):
        print(calc_flux())

#Flam2 is correct, equation A.10, Bessel

'''
<f_lam> = integrate(f_lam * S * wl) / integrate(S * wl)

This is converting f_lam to proportional to photons (since S in pysynphot is indeed a photonic passband), and seeing how many photons it counts. Basically, it is taking the photon-weighted average of f_lam. 

S(λ) is the dimensionless bandpass throughput function, and the division by hν = hc / λ converts the energy flux to a photon flux as is appropriate for photon-counting detectors. (ie, this is independent of the passband function).

For speed, this means we could pre-evaluate the numerator, and scale S such that the integral becomes normalized.
'''


