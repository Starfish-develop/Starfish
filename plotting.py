import numpy as np
import matplotlib.pyplot as plt
import model as m
from matplotlib.ticker import FormatStrFormatter as FSF

base = '/home/ian/Grad/Research/Disks/TRES_spectra/LkCa15/b/LkCa15_2013-10-13_09h37m31s_cb.flux.spec.'
wls = np.load(base + "wls.npy")
fls = np.load(base + "fls.npy")
sigmas = np.load(base + "sigma.npy")
masks = np.load(base + "mask.npy")

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

def plot_wl():
    plt.plot(wl)
    plt.xlabel("Index")
    plt.ylabel(r"Wavelength $\AA$")
    plt.savefig("plots/WAVE_GWOri_solution.png")

def plot_wl_short():
    plt.plot(w)
    plt.xlabel("Index")
    plt.ylabel(r"Wavelength $\AA$")
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
    f_mod = m.model(5900)
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

def three_orders():
    f_mod = m.model(6200, logg=4.0, Av=3.7, orders=m.orders)
    m.global_chi2(f_mod)
    const_coeff = np.load("const_coeff.npy")
    chiR_list = np.load("chiR_list.npy")
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11, 8))
    for i in range(3):
        wl, f_TRES = m.efile_z[m.orders[i]]
        f = f_mod[i]
        p = const_coeff[i]
        ax[0, i].annotate(r"%s $\chi^2_R=$%.2f" % (m.orders[i] + 1, chiR_list[i]), (0.4, 0.85),
                          xycoords="axes fraction")
        ax[0, i].plot(wl, f_TRES, "b")
        ax[0, i].plot(wl, f * Ch(p, domain=[wl[0], wl[-1]])(wl), "r")

        ax[1, i].plot(wl, Ch(p, domain=[wl[0], wl[-1]])(wl))

        ax[2, i].plot(wl, f_TRES - f * Ch(p, domain=[wl[0], wl[-1]])(wl), 'g')
        for k in ax[:, i]:
            k.set_xlim(wl[0], wl[-1])
            k.xaxis.set_major_formatter(FSF("%.0f"))
            k.locator_params(axis='x', nbins=5)
    fig.savefig("plots/three_orders.png")


def one_order():
    f_mod = m.model(6300, logg=4.0, Av=3.7, orders=m.orders)
    m.global_chi2(f_mod)
    const_coeff = np.load("const_coeff.npy")
    chiR_list = np.load("chiR_list.npy")
    chi2_list = np.load("chi2_list.npy")
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(11, 8))
    i = 0
    wl, f_TRES = m.efile_z[m.orders[i]]

    f = f_mod[i]
    p = const_coeff[i]
    ax[0].annotate(r"%s $\chi^2 =$%.1f $\chi^2_R=$%.2f" % (m.orders[i] + 1, chi2_list[i], chiR_list[i]), (0.4, 0.85),
                   xycoords="axes fraction")
    ax[0].plot(wl, f_TRES, "b")
    ax[0].plot(wl, f * Ch(p, domain=[wl[0], wl[-1]])(wl), "r")

    ax[1].plot(wl, Ch(p, domain=[wl[0], wl[-1]])(wl))
    residuals = f_TRES - f * Ch(p, domain=[wl[0], wl[-1]])(wl)

    ax[2].plot(wl, residuals, 'g')
    for k in ax:
        k.set_xlim(wl[0], wl[-1])
        k.xaxis.set_major_formatter(FSF("%.0f"))
        k.locator_params(axis='x', nbins=5)
    fig.savefig("plots/one_order63.png")
    #histogram of residuals
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.hist(residuals, bins=20)
    #plt.show()

def plot_GWOri_all_unnormalized():
    #Load normalized order spectrum
    efile = rechelletxt("GWOri_f") #has structure len = 51, for each order: [wl,fl]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    for i in efile:
        wl, fl = i
        ax.plot(wl, fl)
        ax.xaxis.set_major_formatter(FSF("%.0f"))
        ax.set_xlabel(r"$\lambda\quad[\AA]$")
        ax.set_ylabel("Counts")
    fig.savefig("plots/all_flux_calib.png")

def plot_order_23():
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    wls, fls, fs = m.model_and_data(np.array([5900, 4.0, 6.0, 30, 30, 5.5e27, 0, 0, 0]))
    ax.plot(wls[0], fls[0], "b")
    ax.plot(wls[0], fs[0], "r")
    plt.show()


def main():
    plot_comic_strip()
    pass


if __name__ == "__main__":
    main()

