import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
import model as m
from matplotlib.ticker import FormatStrFormatter as FSF
from scipy.integrate import trapz
from echelle_io import rechelletxt
from numpy.polynomial import Chebyshev as Ch


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


def calc_chi2_grid():
    Ts = np.arange(5700, 6501, 100)
    loggs = np.arange(2.5, 5.1, 0.5)
    TT, GG = np.meshgrid(Ts, loggs)
    chi2 = m.calc_chi2(TT, GG)
    np.save("TT.npy", TT)
    np.save("GG.npy", GG)
    np.save("chi2.npy", chi2)


def plot_T_logg():
    TT, GG, chi2 = np.load("TT.npy"), np.load("GG.npy"), np.load("chi2.npy")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    C = plt.imshow(chi2, vmax=4e3, cmap="gist_stern", origin='lower', aspect="auto",
                   extent=(TT[0, 0], TT[-1, -1], GG[0, 0], GG[-1, -1]), interpolation="nearest")
    fig.colorbar(C)
    #levels = np.arange(6000,6400,50)
    #CS = ax.contour(TT,GG,chi2,levels)
    #ax.clabel(CS, inline=1, fontsize=10)#,fmt = '%2.0f')
    ax.set_xlabel(r"$T_{\rm eff}$")
    ax.set_ylabel("log(g)")
    ax.xaxis.set_major_formatter(FSF("%.0f"))
    plt.show()
    #fig.savefig("chi_contour.eps")


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


def compare_sample():
    #Mg_shift = shift_vz(Mg,v_shift)

    T_list = np.arange(5900, 6301, 100)
    F_list = []
    Chi_list = []
    for T in T_list:
        raw_flux = m.model(T, logg=3.5)
        pref = m.find_pref(raw_flux)
        cal_flux = pref * raw_flux
        F_list.append(cal_flux)
        Chi_list.append(m.chi2(cal_flux))

    fig, axes = plt.subplots(nrows=len(T_list), sharex=True, figsize=(11, 8))

    for i, ax in enumerate(axes):
        ax.plot(m.wl, m.fl, "r")
        ax.plot(m.wl, F_list[i])
        ax.set_title(r"T={:.0f} K $\chi^2 = {:.0f}$".format(T_list[i], Chi_list[i]))

    axes[-1].set_xlabel(r"$\lambda\quad[\AA]$")
    axes[-1].xaxis.set_major_formatter(FSF("%.0f"))
    fig.subplots_adjust(top=0.94, right=0.97, hspace=0.25, left=0.08)
    plt.show()

    #for i in ax:
    #    for j in Mg_shift:
    #        i.axvline(j,ls=":",color="k")
    #plt.savefig("plots/chi2_grid_T.png")


def plot_continuum():
    '''Is this correct??'''
    noise = 1 / np.sqrt(m.cont)
    norm_noise = 0.06 * noise / np.min(noise)
    np.save("sigma.npy", norm_noise)
    plt.plot(m.wl, norm_noise)
    plt.show()


def test_downsample():
    wl, fl = np.loadtxt("GWOri_cn/34.txt", unpack=True)
    f_full = m.load_flux(5900, 3.5)

    #Limit huge file to the necessary order. Even at 4000 ang, 1 angstrom corresponds to 75 km/s. Add in an extra 5
    # angstroms to be sure.
    ind = (m.w_full > 6163.) & (m.w_full < 6164)
    w = m.w_full[ind]
    f = f_full[ind]

    #convolve with stellar broadening (sb)
    #k = m.vsini_ang(np.mean(wl),40.) #stellar rotation kernel centered at order
    #f_sb = m.convolve(f, k)

    #dlam = w[1] - w[0]

    #convolve with filter to resolution of TRES
    #filt = m.gauss_series(dlam,lam0=np.mean(w))
    #f_TRES = m.convolve(f_sb,filt)

    fig, ax = plt.subplots(nrows=2, figsize=(11, 8), sharex=True)
    d1 = m.downsample(w, f_TRES, wl)
    d2 = m.downsample4(w, f_TRES, wl)
    ax[0].plot(wl, d1)
    ax[1].plot(wl, d2)
    ax[2].plot(wl, d1 - d2)
    ax[3].plot(wl, (d1 - d2) / d1)
    ax[-1].set_xlabel(r"$\lambda\quad[\AA]$")
    ax[-1].xaxis.set_major_formatter(FSF("%.0f"))
    fig.subplots_adjust(top=0.94, right=0.97, hspace=0.25, left=0.08)
    plt.show()


def test_bin():
    wl, fl = np.loadtxt("GWOri_cn/34.txt", unpack=True)
    f_full = m.load_flux(5900, 3.5)
    wl = m.shift_vz(wl, 30.)

    indT = (wl > 6163.47) & (wl < 6284)
    wl = wl[indT]
    fl = fl[indT]


    #Determine the TRES bin edges
    len_TRES = len(wl)
    edges = np.empty((len_TRES + 1,), dtype=np.float64)
    difs = np.diff(wl) / 2.
    edges[1:-1] = wl[:-1] + difs
    edges[0] = wl[0] - difs[0]
    edges[-1] = wl[-1] + difs[-1]

    ind = (m.w_full > (edges[0])) & (m.w_full < (0.005 + edges[-1]))

    w = m.w_full[ind]
    f = f_full[ind]
    print(f.dtype)


    #change all to 64 bit to see if this is a rounding error
    w = w.astype(np.float64)
    f = f.astype(np.float64)
    wl = wl.astype(np.float64)

    d1 = m.downsample5(w, f, wl)
    weights = np.ones_like(f)

    #For average between 6237.5 and 6237.65
    #weights[0] = 0.97663176
    #weights[-1] = 0.59715568


    print("Length of f", len(f))
    #For average between 6237.5 and 6250
    weights[0] = 0.69301058
    weights[-1] = 0.72874322

    FTRES = trapz(d1, wl)
    FFULL = trapz(f, w)

    print(w)
    print(weights)

    dmean = np.mean(d1)
    fmean = np.average(f, weights=weights)
    print("Comparing averages", dmean, dmean.dtype, fmean, fmean.dtype)
    print("Comparing total flux", FTRES, FFULL)

    #Yields exactly the same answer for the average between 6237.5 and 6237.65: 
    #Comparing averages 8.45379889267e+14 8.4538e+14

    fig, ax = plt.subplots(nrows=1, figsize=(11, 8))
    #ax[0].plot(wl,fl,"o")
    ax.plot(w, f)
    ax.plot(wl, d1, "ro")
    #for i in ax:
    #ax.set_xlim(6237.5,6238.5)
    #ax.set_ylim(8.1e14,8.6e14)
    plt.show()


def plot_GWOri_merged():
    GW_file = pf.open("GWOri_cnm.fits")
    f = GW_file[0].data
    disp = 0.032920821413025
    w0 = 3850.3823242188
    w = np.arange(len(f)) * disp + w0

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(w, f)
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.set_xlim(3800, 9100)
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
    f_mod = m.model(5900)
    for i in range(10):
        fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(11, 8))
        for j in range(5):
            order = i * 5 + j
            w, f_TRES = m.efile_n[order]
            f = f_mod[order]
            sigma = m.sigmas[order]
            ax[0, j].plot(w, sigma)
            ax[0, j].annotate("%s" % (order + 1), (0.4, 0.85), xycoords="axes fraction")
            ax[1, j].plot(w, f_TRES, "b")
            omask = m.masks[order]
            ax[1, j].plot(w[-omask], f_TRES[-omask], "r")
            #            if len(order_masks) >= 1:
            #                for mask in order_masks:
            #                    start,end = mask
            #                    ind = (w > start) & (w < end)
            #                    ax[1,j].plot(w[ind],f_TRES[ind],"r")
            ax[2, j].plot(w, f)
            for k in ax[:, j]:
                k.set_xlim(w[0], w[-1])
                k.xaxis.set_major_formatter(FSF("%.0f"))
                k.locator_params(axis='x', nbins=5)
        fig.savefig("plots/comic_strips/set%s.png" % (i,))


def plot_tilt():
    wl, fl = np.loadtxt("GWOri_cn/43.txt", unpack=True)
    wl = m.shift_vz(wl, 30.2)
    f = m.model(5900, 3.5, orders=[42])[0]
    z = m.masks[42]
    wl, f, fl, sigma = wl[z], f[z], fl[z], m.sigmas[42][z]

    p = m.find_line(wl, f, fl, sigma)

    fig, ax = plt.subplots(nrows=3, figsize=(11, 8))
    ax[0].plot(wl, fl)
    ax[1].plot(wl, f)
    #ax[1].plot(wl,p[0] + p[1]*wl)
    ax[1].plot(wl, 1.3e15 + -9e10 * wl, 'r')
    ax[2].plot(wl, fl, "b")
    ax[2].plot(wl, f / (p[0] + p[1] * wl), "r")
    plt.show()


def test_global_chi2():
    f_mod = m.model(6000, logg=4.0, Av=3.7, orders=m.orders)
    m.global_chi2(f_mod)
    const_coeff = np.load("const_coeff.npy")
    chiR_list = np.load("chiR_list.npy")
    for i in range(6): #formerly 10
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(11, 8))
        for j in range(5):
            index = i * 5 + j
            wl, f_TRES = m.efile_z[m.orders[index]]
            f = f_mod[index]
            ax[0, j].annotate(r"%s $\chi^2_R=$%.2f" % (m.orders[index] + 1, chiR_list[index]), (0.4, 0.85),
                              xycoords="axes fraction")

            omask = m.masks[m.orders[index]]
            p = const_coeff[index]

            ax[0, j].plot(wl, f_TRES, "b")
            ax[0, j].plot(wl[-omask], f_TRES[-omask], "g")
            ax[0, j].plot(wl, f / p, "r")

            ax[1, j].plot(wl, f_TRES - f / p)

            for k in ax[:, j]:
                k.set_xlim(wl[0], wl[-1])
                k.xaxis.set_major_formatter(FSF("%.0f"))
                k.locator_params(axis='x', nbins=5)
        fig.savefig("plots/comic_strips_tilt/set%s.png" % (i,))


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


def test_chebyshev():
    #coef = np.array([0.3,0.2,0.6])
    coef = np.array([1., 1., 0.])
    myCh = Ch(coef)
    #myCh2 = Ch(coef,domain=[0,3.])
    #xs = np.linspace(0,3.)
    x0 = np.linspace(-1, 1)
    plt.plot(x0, myCh(x0))
    #plt.plot(xs, myCh2(xs))

    plt.show()


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
    #plot_continuum()
    #compare_sample()
    #calc_chi2_grid()
    #plot_T_logg()
    #plot_logg_grid()
    #plot_wl()
    #plot_wl_short()
    #plot_GWOri_all_orders()
    #plot_sigmas()
    #plot_full_model()
    #plot_comic_strip()
    #test_bin()
    #test_downsample()
    #plot_tilt()
    #test_global_chi2()
    #three_orders()
    #one_order()
    #plot_GWOri_all_unnormalized()
    test_chebyshev()
    #plot_order_23()
    pass


if __name__ == "__main__":
    main()

