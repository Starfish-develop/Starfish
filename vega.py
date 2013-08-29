import numpy as np
import pysynphot as S
import matplotlib.pyplot as plt
from echelle_io import rechelletxt
from matplotlib.ticker import FormatStrFormatter as FSF
from scipy.interpolate import interp1d
from model import shift_vz
import asciitable


vega = S.FileSpectrum("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/calspec/alpha_lyr_stis_005.fits")
#feige = S.FileSpectrum("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/calspec/feige34_stis_001.fits")

#plt.plot(vega.wave,vega.flux)
#plt.plot(feige34.wave,feige34.flux)
#plt.xlim(3800,10000)
#plt.show()

#vega_efile = rechelletxt("TRES_spectra/Vega/Vega_2012-04-02_12h50m48s_cb.norm.crop.spec") #has structure len = 51, for each order: [wl,fl]
##vega_sfile = []
#for i in vega_efile:
#    i[1] = i[1]/3.0
#print("Flux normed Vega")
#
#feige_efile = rechelletxt("TRES_spectra/Feige34/Feige34_2012-04-29_03h42m55s_cb.norm.crop.spec")
#for i in feige_efile:
#    i[1] = i[1]/360.0
#print("Flux normed Feige34")
##feige_sfile = []

GWOri_efile = rechelletxt("TRES_spectra/GWOri/11LORI_2012-04-02_02h46m34s_cb.norm.crop.flux.spec")

def plot_vega():
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    vega.convert("abmag")
    ax.plot(vega.wave, vega.flux)
    ax.set_xlim(3800,10000)
    ax.set_ylabel("AB Mag")
    plt.show()

def write_vega_dat():
    ind = (3000 < vega.wave) & (vega.wave < 10000)
    w = vega.wave[ind]
    bpass = np.empty((np.sum(ind),))
    bpass[:-1] = np.diff(w)
    bpass[-1] = bpass[-2]
    vega.convert("abmag")
    print(vega.waveunits)
    print(vega.fluxunits)
    asciitable.write({"w":w,"f":vega.flux,"b":bpass}, "vega.dat",names=["w","f","b"])
    #asciitable.write(sequences,"run_0.dat",names=["j","ratio","accept","scale","pedestal"])
    
def calc_bin(iname,oname):
    data = asciitable.read(iname)
    wl = data['col1']
    ab = data['col2']
    bpass = np.empty((len(wl),))
    bpass[:-1] = np.diff(wl)
    bpass[-1] = bpass[-2]
    asciitable.write({"wl":wl,"ab":ab,"bpass":bpass}, oname, names=["wl","ab","bpass"])

def plot_echelle():
    fig, ax = plt.subplots(nrows=3, figsize=(11,8),sharex=True)

    for i in efile:
        wl,fl = i
        ax[0].plot(wl,fl,"b")

    ax[1].plot(vega.wave, vega.flux)
    ax[1].set_xlim(3800,10000)

    ind = (vega.wave > 3700) & (vega.wave < 10000)
    vega_flux = interp1d(vega.wave[ind],vega.flux[ind],"linear")
    print("Finished interpolation")
    norm = []
    for i in range(51):
        print("Normalizing order %s" % (i,))
        wl,fl = efile[i]
        true_flux = vega_flux(wl)
        X = true_flux/fl
        norm.append(X)
        ax[2].plot(wl,X,"b")

    ax[-1].xaxis.set_major_formatter(FSF("%.0f"))
    ax[-1].set_xlabel(r"$\lambda\quad[\AA]$")
    ax[-1].set_ylabel("Counts")


    plt.show()


def plot_feige():
    fig, ax = plt.subplots(nrows=3, figsize=(11,8),sharex=True)

    for i in feige_efile:
        wl,fl = i
        ax[0].plot(wl,fl,"b")

    ax[1].plot(feige.wave, feige.flux)
    ax[1].set_xlim(3800,10000)

    ind = (feige.wave > 3700) & (feige.wave < 10000)
    feige_flux = interp1d(feige.wave[ind],feige.flux[ind],"linear")
    print("Finished interpolation")
    norm = []
    for i in range(51):
        print("Normalizing order %s" % (i,))
        wl,fl = feige_efile[i]
        true_flux = feige_flux(wl)
        X = true_flux/fl
        norm.append(X)
        ax[2].plot(wl,X,"b")

    ax[-1].xaxis.set_major_formatter(FSF("%.0f"))
    ax[-1].set_xlabel(r"$\lambda\quad[\AA]$")
    ax[-1].set_ylabel("Counts")


    plt.show()

def comp_flux_correction():
    fig, ax = plt.subplots(nrows=2, figsize=(11,8),sharex=True,sharey=True)

    ind_v = (vega.wave > 3700) & (vega.wave < 10000)
    vega_flux = interp1d(vega.wave[ind_v],vega.flux[ind_v],"linear")
    print("Finished interpolation")
    norm = []
    for i in range(51):
        print("Normalizing order %s" % (i,))
        wl,fl = vega_efile[i]
        true_flux = vega_flux(wl)
        X = true_flux/fl
        norm.append(X)
        ax[0].plot(wl,X,"b")
    ax[0].set_title("Vega flux cal")



    ind_f = (feige.wave > 3700) & (feige.wave < 10000)
    feige_flux = interp1d(feige.wave[ind_f],feige.flux[ind_f],"linear")
    print("Finished interpolation")
    norm = []
    for i in range(51):
        print("Normalizing order %s" % (i,))
        wl,fl = feige_efile[i]
        true_flux = feige_flux(wl)
        X = true_flux/fl
        norm.append(X)
        ax[1].plot(wl,X,"b")

    ax[1].set_title("Feige34 flux cal")

    ax[-1].xaxis.set_major_formatter(FSF("%.0f"))
    ax[-1].set_xlabel(r"$\lambda\quad[\AA]$")
    ax[-1].set_ylabel("Counts")
    ax[-1].set_xlim(3800,9300)
    ax[-1].set_ylim(1e-13,5e-12)

    fig.savefig("plots/vega_vs_feige34.png")
    #plt.show()

def write_vega_correction():
    ind_v = (vega.wave > 3700) & (vega.wave < 10000)
    vega_flux = interp1d(vega.wave[ind_v],vega.flux[ind_v],"linear")
    print("Finished interpolation")
    wnorm = np.empty((51,2299))
    fnorm = np.empty((51,2299))
    for i in range(51):
        print("Normalizing order %s" % (i,))
        wl,fl = vega_efile[i]
        true_flux = vega_flux(wl)
        X = true_flux/fl
        wnorm[i] = wl
        fnorm[i] = X
    np.save("vega_wnorm.npy", wnorm)
    np.save("vega_fnorm.npy", fnorm)

def norm_GWOri():
    wnorm = np.load("vega_wnorm.npy")
    fnorm = np.load("vega_fnorm.npy")
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    for i in range(51):
        wl,fl = GWOri_efile[i]
        Xfunc = interp1d(wnorm[i],fnorm[i],"linear",bounds_error=False)
        X = Xfunc(wl)
        fln = fl * X
        ax.plot(wl,fln,"b")

    plt.show()

def plot_GWOri():
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    for i in range(51):
        wl,fl = GWOri_efile[i]
        ax.plot(wl,fl,"b")
        ax.xaxis.set_major_formatter(FSF("%.0f"))
        ax.set_xlabel(r"$\lambda\quad[\AA]$")
        ax.set_ylabel(r"$F_\lambda\quad[\frac{{\rm ergs}}{{\rm cm}^2 /{\rm s}/\AA}]$")

    plt.show()



def main():
    #comp_flux_correction()
    #write_vega_correction()
    #norm_GWOri()
    #write_vega_dat()
    #plot_GWOri()
    #base = "TRES_spectra/FLUX_STANDARDS/"
    #calc_bin(base + "mhr5191.dat",base + "mhr5191b.dat")
    plot_vega()

if __name__=="__main__":
    main()
