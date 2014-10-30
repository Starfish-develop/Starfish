import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from matplotlib.ticker import FormatStrFormatter as FSF
#from scipy.optimize import fsolve

c = 2.99792458e18 #A s^-1


def karray(center, width, res):
    '''Creates a kernel array with an odd number of elements, the central element centered at `center` and spanning
    out to +/- width in steps of resolution. Works similar to arange in that it may or may not get all the way to the
    edge.'''
    neg = np.arange(center - res, center - width, -res)[::-1]
    pos = np.arange(center, center + width, res)
    kar = np.concatenate([neg, pos])
    return kar

#Convolution
@np.vectorize
def gauss_kernel(dlam, V=6.8, lam0=6500.):
    '''V is the FWHM in km/s. lam0 is the central wavelength in A'''
    sigma = V / 2.355 * 1e13 #A/s
    return c_ang / lam0 * 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (c_ang * dlam / lam0) ** 2 / (2. * sigma ** 2))


def plot_gauss():
    lams = np.linspace(-1, 1, num=200)
    plt.plot(lams, gauss_kernel(lams))
    plt.show()


def gauss_series(dlam, V=6.5, lam0=6500.):
    '''sampled from +/- 3sigma at dlam. V is the FWHM in km/s'''
    sigma_l = V / (2.355 * 3e5) * 6500. #A
    wl = karray(0., 4 * sigma_l, dlam)
    gk = gauss_kernel(wl)
    return gk / np.sum(gk)


def plot_gauss_kernel():
    sigma_l = 6.5 / (2.355 * 3e5) * 6500.
    wl = karray(0, 4 * sigma_l, 0.01)
    ys = gauss_series(0.01)
    plt.plot(wl, ys, "o")
    plt.show()

#plot_gauss_kernel()

@np.vectorize
def vsini_kernel(v, vsini, epsilon=0.6):
    '''vsini in km/s. Epsilon is the limb-darkening coefficient, typically 0.6. Formulation uses Eqn 18.14 from Gray,
    The Observation and Analysis of Stellar Photospheres, 3rd Edition.'''
    c1 = 2. * (1 - epsilon) / (np.pi * vsini * (1 - epsilon / 3.))
    c2 = epsilon / (2. * vsini * (1 - epsilon / 3.))
    return c1 * np.sqrt(1. - (v / vsini) ** 2) + c2 * (1. - (v / vsini) ** 2) ** 2


@np.vectorize
def vsini_ang(lam0, vsini, dlam=0.01, epsilon=0.6):
    '''vsini in km/s. Epsilon is the limb-darkening coefficient, typically 0.6. Formulation uses Eqn 18.14 from Gray,
    The Observation and Analysis of Stellar Photospheres, 3rd Edition.'''
    lamL = vsini * 1e13 * lam0 / c
    lam = karray(0, lamL, dlam)
    c1 = 2. * (1 - epsilon) / (np.pi * lamL * (1 - epsilon / 3.))
    c2 = epsilon / (2. * lamL * (1 - epsilon / 3.))
    series = c1 * np.sqrt(1. - (lam / lamL) ** 2) + c2 * (1. - (lam / lamL) ** 2) ** 2
    return series / np.sum(series)


def plot_vsini():
    vs = np.linspace(-20, 20, num=200)
    prof = vsini_kernel(vs, 20.)
    plt.plot(vs, prof)
    plt.show()

#plot_vsini()


def plot_broadened_gauss():
    '''Gaussian centered at 6500 A'''
    #ones stretches from 6300 to 6700 ang
    fig = plt.figure()
    ax = fig.add_subplot(111)
    wl_full = karray(6500, 3, 0.01)
    orig = np.ones_like(wl_full)

    sigma_l = 6.5 / (2.355 * 3e5) * 6500. #A
    wl_gauss = karray(6500., 4 * sigma_l, 0.01)

    ind = (wl_full >= wl_gauss[0]) & (wl_full <= wl_gauss[-1])
    init = 10. - gauss_series(0.01)
    norm = init / 10.
    orig[ind] = norm

    k20 = vsini_ang(6500., 20.)
    f20 = convolve(orig, k20)

    k40 = vsini_ang(6500., 40.)
    f40 = convolve(orig, k40)

    k60 = vsini_ang(6500., 60.)
    f60 = convolve(orig, k60)

    k80 = vsini_ang(6500., 80.)
    f80 = convolve(orig, k80)

    ax.plot(wl_full, orig)
    ax.plot(wl_full, f20)
    ax.plot(wl_full, f40)
    ax.plot(wl_full, f60)
    ax.plot(wl_full, f80)
    ax.xaxis.set_major_formatter(FSF("%.0f"))

    #ax.set_ylim(0.3,1.05)
    ax.set_ylabel(r"$F/F_c$")
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    plt.show()


def main():
    plot_broadened_gauss()


if __name__ == "__main__":
    main()


