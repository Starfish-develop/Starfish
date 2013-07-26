import numpy as np
import matplotlib.pyplot as plt
import model as m
from matplotlib.ticker import FormatStrFormatter as FSF

def plot_logg_grid():
    fig, ax = plt.subplots(nrows=13,sharex=True,sharey=True,figsize=(11,8))
    #fig = plt.figure(figsize=(11,8))
    #ax = fig.add_subplot(111)
    ax[0].plot(m.wl,m.fl)
    ax[0].set_ylabel("GW Ori")
    for i,j in enumerate(np.arange(0.5,6.1,0.5)):
        f = m.model(5900,-30,logg=j)
        ax[i+1].plot(m.wl, f/f[0])
        ax[i+1].set_ylabel("%.2f" % j)
    ax[-1].locator_params(axis='y',nbins=3)
    ax[-1].set_xlabel(r"$\lambda\quad[\AA]$")
    ax[-1].xaxis.set_major_formatter(FSF("%.0f"))
    fig.subplots_adjust(top=0.96,right=0.96)
    plt.show()

def calc_chi2_grid():
    Ts = np.arange(5700,6501,100)
    loggs = np.arange(2.5,5.1,0.5)
    TT,GG = np.meshgrid(Ts,loggs)
    chi2 = m.calc_chi2(TT,GG)
    np.save("TT.npy",TT)
    np.save("GG.npy",GG)
    np.save("chi2.npy",chi2)


def plot_T_logg():
    TT,GG,chi2 = np.load("TT.npy"),np.load("GG.npy"),np.load("chi2.npy")

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    C = plt.imshow(chi2,vmax=4e3,cmap="gist_stern",origin='lower',aspect="auto",extent=(TT[0,0],TT[-1,-1],GG[0,0],GG[-1,-1]),interpolation="nearest")
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

def compare_sample():
    #Mg_shift = shift_vz(Mg,v_shift)

    T_list = np.arange(5900,6301,100)
    F_list = []
    Chi_list = []
    for T in T_list:
        raw_flux = m.model(T,logg=3.5)
        pref = m.find_pref(raw_flux)
        cal_flux = pref * raw_flux
        F_list.append(cal_flux)
        Chi_list.append(m.chi2(cal_flux))
    
    fig, axes = plt.subplots(nrows=len(T_list),sharex=True,figsize=(11,8))

    for i,ax in enumerate(axes):
        ax.plot(m.wl,m.fl,"r")
        ax.plot(m.wl,F_list[i])
        ax.set_title(r"T={:.0f} K $\chi^2 = {:.0f}$".format(T_list[i],Chi_list[i]))

    axes[-1].set_xlabel(r"$\lambda\quad[\AA]$")
    axes[-1].xaxis.set_major_formatter(FSF("%.0f"))
    fig.subplots_adjust(top=0.94,right=0.97,hspace=0.25,left=0.08)
    plt.show()

    #for i in ax:
    #    for j in Mg_shift:
    #        i.axvline(j,ls=":",color="k")
    #plt.savefig("plots/chi2_grid_T.png")

def plot_continuum():
    '''Is this correct??'''
    noise = 1/np.sqrt(m.cont)
    norm_noise = 0.06 * noise/np.min(noise)
    np.save("sigma.npy",norm_noise)
    plt.plot(m.wl,norm_noise)
    plt.show()
    

def main():
    #plot_continuum()
    compare_sample()
    #calc_chi2_grid()
    #plot_T_logg()
    #plot_logg_grid()
    #plot_wl()
    #plot_wl_short()
    pass

if __name__=="__main__":
    main()

