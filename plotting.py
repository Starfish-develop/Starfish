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

def line_classes():
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(6,4.))
    #need to set config.yaml['orders'] = [22,23], WASP-14
    fl23, fl24 = m.fls
    sig23, sig24 = m.sigmas

    wlsz23 = np.load("residuals/wlsz23.npy")[0]
    f23 = np.load("residuals/fl23.npy")[0]
    wlsz24 = np.load("residuals/wlsz24.npy")[0]
    f24 = np.load("residuals/fl24.npy")[0]

    #import matplotlib
    #font = {'size' : 8}

    #matplotlib.rc('font', **font)
    #matplotlib.rc('labelsize', **font)
    r23 = (fl23 - f23)/sig23
    r24 = (fl24 - f24)/sig24
    fl23 /= 2e-13
    fl24 /= 2e-13
    f23 /= 2e-13
    f24 /= 2e-13


    ax[0,0].plot(wlsz23, fl23, "b", label="data")
    ax[0,0].plot(wlsz23, f23, "r", label="model")

    ax[1,0].plot(wlsz23, r23, "g")
    ax[0,0].set_xlim(5136.4, 5140.4)
    ax[1,0].set_xlim(5136.4, 5140.4)
    ax[1,0].set_ylim(-4,4)

    ax[0,1].plot(wlsz23, fl23, "b")
    ax[0,1].plot(wlsz23, f23, "r")
    ax[1,1].plot(wlsz23, r23, "g")
    ax[0,1].set_xlim(5188, 5189.5)
    ax[1,1].set_xlim(5188, 5189.5)
    #ax[1,1].set_ylim(-4,4)

    ax[0,2].plot(wlsz24, fl24, "b", label='data')
    ax[0,2].plot(wlsz24, f24, "r", label='model')
    ax[0,2].legend(loc="lower center", prop={'size':10})

    ax[1,2].plot(wlsz24, r24, "g")
    ax[0,2].set_xlim(5258, 5260)
    ax[1,2].set_xlim(5258, 5260)

    ax[0,3].plot(wlsz24, fl24, "b")
    ax[0,3].plot(wlsz24, f24, "r")
    ax[1,3].plot(wlsz24, r24, "g")
    ax[0,3].set_xlim(5260, 5271)
    ax[1,3].set_xlim(5260, 5271)
    ax[1,3].set_ylim(-15, 15)


    for j in range(4):
        labels = ax[1,j].get_xticklabels()
        for label in labels:
            label.set_rotation(60)

    ax[0,0].set_ylabel(r"$\propto f_\lambda$")
    ax[1,0].set_ylabel(r"Residuals$/\sigma_P$")

    for i in range(2):
        for j in range(4):
            ax[i,j].xaxis.set_major_formatter(FSF("%.0f"))
            ax[i,j].xaxis.set_major_locator(MultipleLocator(1.))
            ax[i,j].tick_params(axis='both', which='major', labelsize=10)

    for i in [0,1]:
        for j in [1,2]:
            ax[i,j].xaxis.set_major_formatter(FSF("%.1f"))
            ax[i,j].xaxis.set_major_locator(MultipleLocator(0.5))

    for i in [0,1]:
        ax[i,3].xaxis.set_major_formatter(FSF("%.0f"))
        ax[i,3].xaxis.set_major_locator(MultipleLocator(2))

    class_label = ["0", "I", "II", "III"]
    for j in range(4):
        ax[0,j].set_title(class_label[j])
        ax[0,j].xaxis.set_ticklabels([])
        ax[0,j].set_ylim(0.25, 1.15)
        if j != 0:
            ax[0,j].yaxis.set_ticklabels([])

    fig.subplots_adjust(left=0.09, right=0.99, top=0.94, bottom=0.18, hspace=0.1, wspace=0.27)
    fig.text(0.48, 0.02, r"$\lambda$ (\AA)")
    fig.savefig("plots/badlines.eps")
    #plt.show()


    pass

def main():
    #identify_lines([5083, 5086], [5900, 6000], [3.0, 3.5], ["-1.0", "-0.5"])
    #compare_kurucz()
    #plot_residuals()
    #plot_line_residuals()
    #p24 = np.load("residuals/p24.npy")
    #p24 = np.load("p24.npy")
    #wlsz, refluxed, k, flatchain = m.model_p(p24)
    #np.save("residuals/wlsz24.npy", wlsz)
    #np.save("residuals/fl24.npy", refluxed)
    line_classes()

    pass



if __name__ == "__main__":
    main()

