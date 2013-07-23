import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
from echelle_io import rechelletxt
from scipy.interpolate import interp1d
from scipy.integrate import quad,trapz
from scipy.ndimage.filters import convolve
from kernel import gauss_series, vsini_ang
from matplotlib.ticker import FormatStrFormatter as FSF

c_ang = 2.99792458e18 #A s^-1
c_kms = 2.99792458e5 #km s^-1

def load_file(temp,logg):
    fname="HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{temp:0>5d}-{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(temp=temp,logg=logg)
    print(fname)

load_file(5700,4.5)
load_file(5800,3.5)


#flux_file = pf.open("HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte05500-0.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
#flux_1 = pf.open("HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte05700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
#flux_2 = pf.open("HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
#wl_file = pf.open("WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
#
#f_pure1 = flux_1[0].data
#f_pure2 = flux_2[0].data
#
##f_pure = flux_file[0].data
#w = wl_file[0].data
#
#wl,fl = np.loadtxt("GWOri_c/23.txt",unpack=True)
#wl_n,fl_n = np.loadtxt("GWOri_cn/23.txt",unpack=True)

#efile = rechelletxt()
#use order 36 for all testing
#wl,fl = efile[22]

#Limit huge file to necessary range
#ind = (w > (wl[0] - 10.)) & (w < (wl[-1] + 10))

#f_pure = f_pure[ind]/10**13
#f_pure1 = f_pure1[ind]/10**13
#f_pure2 = f_pure2[ind]/10**13
#w = w[ind]

#flux_file.close()
#wl_file.close()


@np.vectorize
def shift_vz(lam_source, v):
    '''Given the source wavelength, lam_sounce, return the observed wavelength based upon a velocity v in km/s. Negative velocities are towards the observer (blueshift).'''
    lam_observe = lam_source * np.sqrt((c_kms + v)/(c_kms - v))
    return lam_observe

#convolve with stellar broadening
#k = vsini_ang(5187.,40.)
#f_broad = convolve(f_pure, k)

#convolve with filter to resolution of TRES
#filt = gauss_series(0.01,lam0=5187.)
#f_TRES1 = convolve(f_pure1,filt)
#f_TRES2 = convolve(f_pure2,filt)
#f = f_TRES



#model = interp1d(w,f,kind="linear")
#print("Interpolation done")

#Determine the bin edges
#edges = np.empty((len(wl)+1,))
#difs = np.diff(wl)/2.
#edges[1:-1] = wl[:-1] + difs
#edges[0] = wl[0] - difs[0]
#edges[-1] = wl[-1] + difs[-1]
#b0s = edges[:-1]
#b1s = edges[1:]

@np.vectorize
def avg_bin(bin0,bin1):
    return quad(model,bin0,bin1)[0]/(bin1 - bin0)

@np.vectorize
def avg_bin2(bin0,bin1):
    xs = np.linspace(bin0,bin1,num=20)
    ys = model(xs)
    return trapz(ys,xs)/(bin1-bin0)

@np.vectorize
def avg_bin3(bin0,bin1):
    mdl_ind = (w > bin0) & (w < bin1)
    wave = np.empty((np.sum(mdl_ind)+2,))
    flux = np.empty((np.sum(mdl_ind)+2,))
    wave[0] = bin0
    wave[-1] = bin1
    flux[0] = model(bin0)
    flux[-1] = model(bin1)
    wave[1:-1] = w[mdl_ind]
    flux[1:-1] = f[mdl_ind]
    return trapz(flux,wave)/(bin1-bin0)

#print("Beginning averaging")
#samp = avg_bin3(b0s,b1s)

def plot_check():
    ys = np.ones((11,))
    plt.plot(wl[-10:],ys[-10:],"o")
    plt.plot(edges[-11:],ys,"o")
    plt.show()

def compare_sample():
    fig, ax = plt.subplots(nrows=5,sharex=True,figsize=(11,8))
    ax[0].plot(wl_n,fl_n)
    ax[0].set_title("GW Ori normalized, order 23")
    ax[1].plot(w,f_pure)
    ax[1].set_title("PHOENIX T=5500 K")

    ax[2].plot(w,f_broad)
    ax[2].set_title(r"PHOENIX, $v \sin i = 40$ km/s")

    ax[3].plot(w,f_TRES)
    ax[3].set_title("PHOENIX, convolved 6.5 km/s")


    ax[4].plot(wl,samp)
    ax[4].set_title("PHOENIX, convolved, downsampled")
    #ax[0].set_xlim(wl[0],wl[-1])
    #ax[0].set_xlim(6445,6461)
    ax[4].set_xlabel(r"$\lambda\quad[\AA]$")
    ax[4].xaxis.set_major_formatter(FSF("%.0f"))
    fig.subplots_adjust(top=0.97,right=0.97,hspace=0.25,left=0.08)
    plt.show()

#compare_sample()

def compare_kurucz():
    wl,fl = np.loadtxt("kurucz.txt",unpack=True)
    wl = 10.**wl
    fig, ax = plt.subplots(nrows=4,sharex=True,figsize=(8,8))
    ax[0].plot(wl,fl)
    ax[0].set_title("Kurucz T=5750 K, convolved to 6.5 km/s")
    ax[1].plot(w,f_TRES1)
    ax[1].set_title("PHOENIX T=5700 K, convolved 6.5 km/s")
    ax[2].plot(w,f_TRES2)
    ax[2].set_title("PHOENIX T=5800 K, convolved 6.5 km/s")
    ax[3].plot(wl_n,fl_n)
    ax[3].set_title("GW Ori normalized, order 23")
    ax[-1].xaxis.set_major_formatter(FSF("%.0f"))
    ax[-1].set_xlim(5170,5195)
    ax[-1].set_xlabel(r"$\lambda\quad[\AA]$")
    plt.show()

#compare_kurucz()
