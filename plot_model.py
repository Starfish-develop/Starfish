import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
from scipy.signal import decimate



flux_file = pf.open("HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte04000-0.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
wl_file = pf.open("WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")

#f2 = pf.open("spec_lres.fits")


flux = flux_file[0].data
wl = wl_file[0].data

ind = (wl > 3000) & (wl < 10000)

f = flux[ind]
w = wl[ind]

def plot_wl():
    plt.plot(wl)
    plt.xlabel("Index")
    plt.ylabel(r"Wavelength $\AA$")
    plt.savefig("plots/WAVE_solution.png")

def plot_wl_short():
    plt.plot(w)
    plt.xlabel("Index")
    plt.ylabel(r"Wavelength $\AA$")
    plt.show()
    #plt.savefig("plots/WAVE_solution_trunc.png")

def plot_model_short():
    plt.plot(w,f)
    plt.xlabel(r"$\lambda\quad[\AA]$")
    plt.ylabel(r"Flux [$\frac{{\rm erg}}{{\rm s/}{\rm cm}^2/{\rm cm}}$]")
    plt.subplots_adjust(left=0.2)
    plt.show()
    #plt.savefig("plots/2300_trunc.png")


def main():
    #plot_wl()
    #plot_wl_short()
    plot_model_short()
    pass

if __name__=="__main__":
    main()

#How to bin correctly?

#Degrade the spectrum
#filt = np.ones((1000,))
#df = np.convolve(f,filt,"valid")
#dw = np.convolve(w,filt,"valid")

#plt.plot(dw,df)
#plt.show()
