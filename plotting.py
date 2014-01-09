import numpy as np
import matplotlib.pyplot as plt
import model as m
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MultipleLocator
import yaml
import PHOENIX_tools as pt
import h5py
from astropy.io import ascii
import plot_MCMC as pltMC
from scipy.optimize import fmin

f = open('config.yaml')
config = yaml.load(f)
f.close()

c_ang = 2.99792458e18 #A s^-1

base = 'data/' + config['dataset']
wls = np.load(base + ".wls.npy")
fls = np.load(base + ".fls.npy")
#fls_true = np.load(base + ".true.fls.npy")
sigmas = np.load(base + ".sigma.npy")
masks = np.load(base + ".mask.npy")

#k.xaxis.set_major_formatter(FSF("%.0f"))
#k.locator_params(axis='x', nbins=5)

def plot_logg_grid():
    fig, ax = plt.subplots(nrows=13, sharex=True, sharey=True, figsize=(11, 8))
    #fig = plt.figure(figsize=(11,8))
    #ax = fig.add_subplot(111)
    ax[0].plot(m.wl, m.fl)
    ax[0].set_ylabel("GW Ori")
    for i, j in enumerate(np.arange(0.5, 6.1, 0.5)):
        f = m.model(5900, -30, logg=j)
        ax[i + 1].plot(m.wl, f / f[0])
        ax[i + 1].set_ylabel("%.2f" % j)
    ax[-1].locator_params(axis='y', nbins=3)
    ax[-1].set_xlabel(r"$\lambda\quad[\AA]$")
    ax[-1].xaxis.set_major_formatter(FSF("%.0f"))
    fig.subplots_adjust(top=0.96, right=0.96)
    plt.show()

def compare_kurucz():
    wg_full = np.load('wave_grids/PHOENIX_2kms_air.npy')
    wg = np.load("wave_grids/kurucz_2kms_air.npy")

    ind = (wg_full >= wg[0]) & (wg_full <= wg[-1])

    kurucz = m.load_hdf5_spectrum(6000, 4.0, 0.0, "kurucz", "LIB_kurucz_2kms_air.hdf5")
    BTSettl = m.load_hdf5_spectrum(6000, 4.0, 0.0, "BTSettl", "LIB_BTSettl_2kms_air.hdf5")[ind]
    PHOENIX = m.load_hdf5_spectrum(6000, 4.0, 0.0, "PHOENIX", "LIB_PHOENIX_2kms_air.hdf5")[ind]

    #normalize avg value to 1
    kurucz *= c_ang/wg**2
    kurucz = kurucz/np.average(kurucz)
    BTSettl = BTSettl/np.average(BTSettl)
    PHOENIX = PHOENIX/np.average(PHOENIX)

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(9,6))

    ax[0].plot(wg, kurucz, label="Kurucz")
    ax[0].plot(wg, PHOENIX, label="Husser")
    ax[0].legend()

    residuals = (kurucz - PHOENIX)/kurucz
    ax[1].plot(wg, residuals, label="(Kurucz-Husser)/Kurucz")
    ax[1].legend()
    line_list = pltMC.return_lines(wg, residuals, sigma=0.3, tol=0.2)

    offsets = np.linspace(-0.4, 0.4, num=10)
    off_counter = 0
    for line, label in line_list:
        ax[1].axvline(line, color='k', lw=0.1)
        ax[1].annotate("%s" % label, (line, 0.5 + offsets[off_counter % 10]), xycoords=('data', 'axes fraction'), rotation='vertical', ha='center', va='center', size=12)
        off_counter += 1

    ax[1].xaxis.set_major_formatter(FSF("%.2f"))
    #ax[1].xaxis.set_major_locator(MultipleLocator(.))
    #ax[1].xaxis.set_minor_locator(MultipleLocator(1.))

    plt.show()
    plt.hist(residuals, bins=np.linspace(-0.6,0.6,num=50),log=True)
    plt.xlabel(r"$\sigma$")
    #plt.ylim(0,500)
    plt.savefig("plots/kurucz_husser_residuals_log.png")
    plt.show()

def plot_GWOri_all_orders():
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(11, 8))
    for i in range(5):
        for j in range(5):
            order = i * 5 + j + 25
            w, f = m.efile_n[order]
            ax[i, j].plot(w, f)
            ax[i, j].xaxis.set_major_formatter(FSF("%.0f"))
            ax[i, j].locator_params(axis='x', nbins=5)
            ax[i, j].set_xlim(w[0], w[-1])
            ax[i, j].annotate("%s" % (order + 1), (0.1, 0.85), xycoords="axes fraction")
    fig.subplots_adjust(left=0.04, bottom=0.04, top=0.97, right=0.97)
    ax[4, 0].set_xlabel(r"$\lambda\quad[\AA]$")
    plt.show()

def plot_sigmas():
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(11, 8))
    for i in range(5):
        for j in range(5):
            order = i * 5 + j #+ 25
            w, f = m.efile_n[order]
            sigma = m.sigmas[order]
            print(len(w), len(sigma))
            ax[i, j].plot(w, sigma)
            ax[i, j].xaxis.set_major_formatter(FSF("%.0f"))
            ax[i, j].locator_params(axis='x', nbins=5)
            ax[i, j].set_xlim(w[0], w[-1])
            ax[i, j].annotate("%s" % (order + 1), (0.1, 0.85), xycoords="axes fraction")
    fig.subplots_adjust(left=0.04, bottom=0.04, top=0.97, right=0.97)
    ax[4, 0].set_xlabel(r"$\lambda\quad[\AA]$")
    fig.savefig("plots/sigmas25.png")

def plot_full_model():
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(11, 8))
    f_mod = m.model(6200,3.5,0.0,)
    for i in range(5):
        for j in range(5):
            order = i * 5 + j + 25
            w, f = m.efile_n[order]
            f = f_mod[order]
            ax[i, j].plot(w, f)
            ax[i, j].xaxis.set_major_formatter(FSF("%.0f"))
            ax[i, j].locator_params(axis='x', nbins=5)
            ax[i, j].set_xlim(w[0], w[-1])
            ax[i, j].annotate("%s" % (order + 1), (0.1, 0.85), xycoords="axes fraction")
    fig.subplots_adjust(left=0.04, bottom=0.04, top=0.97, right=0.97)
    ax[4, 0].set_xlabel(r"$\lambda\quad[\AA]$")
    plt.show()

def plot_comic_strip():
    for i in range(10):
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(11, 8))
        for j in range(5):
            order = i * 5 + j
            wl = wls[order]
            fl = fls[order]
            sigma = sigmas[order]
            mask = masks[order]

            ax[0, j].plot(wl, sigma)
            ax[0, j].annotate("%s" % (order + 1), (0.4, 0.85), xycoords="axes fraction")

            ax[1, j].fill_between(wl, fl - sigma, fl + sigma, color="0.5",alpha=0.8)
            ax[1, j].plot(wl, fl, "b")

            ax[1, j].plot(wl[-mask], fl[-mask], "r")

            for k in ax[:, j]:
                k.set_xlim(wl[0], wl[-1])
                k.xaxis.set_major_formatter(FSF("%.0f"))
                k.locator_params(axis='x', nbins=5)
        fig.savefig("plots/comic_strips/set%s.png" % (i,))

def identify_lines(wi, temp, logg, Z):
    lines = ascii.read("linelist.dat", Reader=ascii.FixedWidth, col_starts=[3,17], col_ends=[16,27],
                       converters={'line': [ascii.convert_numpy(np.float)],
                                   'element': [ascii.convert_numpy(np.str)]})
    print(lines.dtype)

    wl = pt.w
    ind = (wl >= wi[0]) & (wl <= wi[1])
    wl = wl[ind]

    combinations = [[temp[0], logg[0], Z[0]],
                    [temp[0], logg[0], Z[1]],
                    [temp[0], logg[1], Z[0]],
                    [temp[0], logg[1], Z[1]],
                    [temp[1], logg[0], Z[0]],
                    [temp[1], logg[0], Z[1]],
                    [temp[1], logg[1], Z[0]],
                    [temp[1], logg[1], Z[1]]]

    #[print(comb) for comb in combinations]
    fluxes = [pt.load_flux_full(*comb, norm=True)[pt.ind][ind] for comb in combinations]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    ax.xaxis.set_major_locator(MultipleLocator(1.))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))

    ind2 = (lines['line'] >= wi[0]) & (lines['line'] <= wi[1])
    for line, label in lines[ind2]:
        ax.axvline(float(line), color="0.5")
        ax.annotate(label, (line, 0.9), xycoords=('data', 'axes fraction'), rotation='vertical', ha='center', va='center')

    for i, fl in enumerate(fluxes):
        ax.plot(wl, fl, label="%s %s %s" % tuple(combinations[i]))

    ax.legend()
    plt.show()
    pass

def gaussian_abs(p):
    func = lambda x: p[0]/(np.sqrt(2 * np.pi) * p[2]) * np.exp(-0.5 * np.abs(x - p[1])/p[2])
    return func

def gaussian(p):
    func = lambda x: p[0]/(np.sqrt(2 * np.pi) * p[2]) * np.exp(-0.5 * (x - p[1])**2/p[2]**2)
    return func

def lorentzian(p):
    func = lambda x: p[0] / (np.pi * p[2] * np.sqrt(2) * (1 + (x - p[1])**2/(2 * p[2]**2)))
    return func

def combo(p):
    func = lambda x: p[0] * p[2]/np.sqrt(2 * np.pi) * ( (1 - np.exp(-0.5 * (x - p[1])**2/p[2]**2 ))/(x - p[1])**2 )
    return func

def mixed(p):
    func = lambda x: p[0] * (np.exp(-0.5 * x**2/p[2]**2) + p[1] * np.exp(-np.abs(x)/p[3]))
    return func

def plot_residuals():
    residuals = 3 * np.load("residuals/residuals25.npy")
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    n, bins = np.histogram(residuals, bins=40)
    n = n/np.max(n)
    bin_centers = (bins[:-1] + bins[1:])/2
    var = n.copy()
    var[var == 0] = 1.

    abs_func = lambda p: np.sum((n - gaussian_abs(p)(bin_centers))**2/var)
    gauss_func = lambda p: np.sum((n - gaussian(p)(bin_centers))**2/var)
    lorentz_func = lambda p: np.sum((n - lorentzian(p)(bin_centers))**2/var)
    combo_func = lambda p: np.sum((n - combo(p)(bin_centers))**2/var)
    mixed_func = lambda p: np.sum((n - mixed(p)(bin_centers))**2/var)


    #abs_param = fmin(abs_func, [1, 0, 3])
    #abs = gaussian_abs(abs_param)
    #
    #gparam = fmin(gauss_func, [1, 0, 3])
    #gauss = gaussian(gparam)
    #
    #lparam = fmin(lorentz_func, [1, 0, 3])
    #lorentz = lorentzian(lparam)
    #
    #cparam = fmin(combo_func, [1, 0, 3])
    #comb = combo(cparam)

    mparam = fmin(mixed_func, [1, 0.3, 3, 3])
    print("Mixture parameters", mparam)
    mix = mixed(mparam)

    xs = np.linspace(-15, 20, num=100)
    ax.plot(bin_centers, n, "o")
    #ax.plot(xs, abs(xs), label="Exp")
    #ax.plot(xs, gauss(xs), label="Gaussian")
    #ax.plot(xs, lorentz(xs), label="Lorentzian")
    #ax.plot(xs, comb(xs), label="Sivia")
    ax.plot(xs, mix(xs), label="Mixed")
    ax.set_ylabel("Residuals")
    ax.set_xlabel(r"$\sigma$")
    ax.legend()

    #fig.savefig("plots/residuals.png")
    plt.show()

def plot_line_residuals():
    residuals = 3 * np.load("residuals/residuals24.npy")
    wl = wls[23]
    gauss1 = gaussian([4.5, 5245.91, 0.08])
    gauss2 = gaussian([3.0, 5292.86, 0.08])
    gauss3 = gaussian([1.5, 5294.35, 0.08])
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    ax.plot(wl, residuals)
    ax.plot(wl, gauss1(wl))
    ax.plot(wl, gauss2(wl))
    ax.plot(wl, gauss3(wl))
    #plt.xlim(5245.4, 5246.4)

    plt.show()

def main():
    #identify_lines([5083, 5086], [5900, 6000], [3.0, 3.5], ["-1.0", "-0.5"])
    #compare_kurucz()
    #plot_residuals()
    plot_line_residuals()
    pass



if __name__ == "__main__":
    main()

