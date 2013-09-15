#!/usr/bin/env python2

import pysynphot as S
import pyfits as pf
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from PHOENIX_tools import load_flux_full,w_full
from deredden import deredden

c_ang = 2.99792458e18 #A s^-1

#data for GWOri
#wl [microns], f_nu [Jy], sys_err (as fraction of f_nu)
data = np.array([[0.36,  6.482e-02, 0.12],
                [0.44, 2.451e-01, 0.30],
                [0.55, 4.103e-01, 0.08],
                [0.64,  7.844e-01, 0.11],
                [0.79, 9.174e-01, 0.07]])

wl = data[:,0]*1e4 #angstroms
f_nu = data[:,1] * 1e-23 #ergs/cm^2/s/Hz
f_nu_err = f_nu * data[:,2] #ergs/cm^2/s/Hz

#Convert to f_lambda
f_lam = f_nu * c_ang/wl**2
f_lam_err = f_nu_err * c_ang/wl**2

filters = ["U","B","V","R","I"]

ind = (w_full > 2000) & (w_full < 40000)
ww = w_full[ind]


ff = load_flux_full(5900,3.5)[ind]*2e-28
#redden spectrum
#red = ff/deredden(ww,1.5,mags=False)

sp = S.ArraySpectrum(wave=ww, flux=ff, waveunits='angstrom', fluxunits='flam', name='T5900K')

#sdss_u=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/sdss_u_005_syn.fits")
#sdss_g=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/sdss_g_005_syn.fits")
#sdss_r=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/sdss_r_005_syn.fits")
#sdss_i=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/sdss_i_005_syn.fits")
#sdss_z=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/sdss_z_005_syn.fits")

U=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/landolt_u_004_syn.fits")
B=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/landolt_b_004_syn.fits")
V=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/landolt_v_004_syn.fits")
R=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/landolt_r_004_syn.fits")
I=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/landolt_i_004_syn.fits")

#H = S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/bessell_h_004_syn.fits")
#J = S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/bessell_j_003_syn.fits")
#K = S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/bessell_k_003_syn.fits")

#obs = S.Observation(sp,sdss_u)
#print("Filter = {name}\tAB = {AB:.3f}\tVega = {Vega:.3f}".format(name="i",AB=obs.effstim("abmag"),Vega=obs.effstim("vegamag")))

pfilts = [U,B,V,R,I]

def calc_fluxes():
    return np.array([S.Observation(sp,i).effstim("flam") for i in pfilts])

#fluxes = calc_fluxes()

def plot_SED():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(wl, f_lam, yerr=f_lam_err,ls="",fmt="o")
    ax.plot(wl, fluxes, "o")
    ax.set_xlabel(r"$\lambda$ [\AA]")
    ax.set_ylabel(r"$F_\lambda$ $\left [\frac{{\rm erg}}{{\rm s} \cdot {\rm cm}^2 \cdot {\rm \AA}}  \right ]$")
    plt.show()


def main():
    print(calc_fluxes())
    #plot_SED()

if __name__=="__main__":
    main()
