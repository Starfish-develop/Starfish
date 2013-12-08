import numpy as np
import matplotlib.pyplot as plt
import model as m
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MultipleLocator
import yaml
import PHOENIX_tools as pt
import h5py
from astropy.io import ascii

f = open('config.yaml')
config = yaml.load(f)
f.close()

base = 'data/' + config['dataset']
wls = np.load(base + ".wls.npy")
fls = np.load(base + ".fls.npy")
fls_true = np.load(base + ".true.fls.npy")
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
    wl, fl = np.loadtxt("kurucz.txt", unpack=True)
    wl = 10. ** wl
    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))
    ax[0].plot(wl, fl)
    ax[0].set_title("Kurucz T=5750 K, convolved to 6.5 km/s")
    ax[1].plot(w, f_TRES1)
    ax[1].set_title("PHOENIX T=5700 K, convolved 6.5 km/s")
    ax[2].plot(w, f_TRES2)
    ax[2].set_title("PHOENIX T=5800 K, convolved 6.5 km/s")
    ax[3].plot(wl_n, fl_n)
    ax[3].set_title("GW Ori normalized, order 23")
    ax[-1].xaxis.set_major_formatter(FSF("%.0f"))
    ax[-1].set_xlim(5170, 5195)
    ax[-1].set_xlabel(r"$\lambda\quad[\AA]$")
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

def main():
    identify_lines([5083, 5086], [5900, 6000], [3.0, 3.5], ["-1.0", "-0.5"])
    pass


if __name__ == "__main__":
    main()

