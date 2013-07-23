#!/usr/bin/env python2

import pysynphot as S
import pyfits as pf
from collections import OrderedDict

# My own synthetic photometry package to test pysynphot


flux_file = pf.open("HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte05700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
wl_file = pf.open("WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")

f_pure = flux_file[0].data
w = wl_file[0].data

#Limit huge file to necessary range
ind = (w > (2000.)) & (w < (40000.))

#f_pure = f_pure[ind]/10**13
ff = f_pure[ind]*5e-28
ww = w[ind]

sp = S.ArraySpectrum(wave=ww, flux=ff, waveunits='angstrom', fluxunits='flam', name='T5700K')

sdss_u=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/sdss_u_005_syn.fits")
sdss_g=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/sdss_g_005_syn.fits")
sdss_r=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/sdss_r_005_syn.fits")
sdss_i=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/sdss_i_005_syn.fits")
sdss_z=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/sdss_z_005_syn.fits")


B=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/landolt_b_004_syn.fits")
V=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/landolt_v_004_syn.fits")
R=S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/landolt_r_004_syn.fits")
H = S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/bessell_h_004_syn.fits")
J = S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/bessell_j_003_syn.fits")
K = S.FileBandpass("/home/ian/.builds/stsci_python-2.12/pysynphot/cdbs/comp/nonhst/bessell_k_003_syn.fits")

#obs = S.Observation(sp,sdss_u)
#print("Filter = {name}\tAB = {AB:.3f}\tVega = {Vega:.3f}".format(name="i",AB=obs.effstim("abmag"),Vega=obs.effstim("vegamag")))

filts = OrderedDict([("u",sdss_u),("g",sdss_g),("r",sdss_r),("i",sdss_i),("z",sdss_z),("B",B),("V",V),("R",R),("H",H),("J",J),("K",K)])
#print(filts.keys())
#
for i in filts.keys():
    obs = S.Observation(sp,filts[i])
    print("Filter = {name}\tAB = {AB:.3f}\tVega = {Vega:.3f}".format(name=i,AB=obs.effstim("abmag"),Vega=obs.effstim("vegamag")))
