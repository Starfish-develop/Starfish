import matplotlib.pyplot as plt
import numpy as np

import os, sys

parentdir = "/home/ian/Grad/Research/Disks/StellarSpectra/"
sys.path.insert(0, parentdir)
from echelle_io import rechelletxt

from matplotlib.ticker import FormatStrFormatter as FSF
import asciitable

c_ang = 2.99792458e18 #A s^-1

GW = rechelletxt("../GWOri/11LORI_2012-04-02_02h46m34s_cb.norm.crop.flux.spec")

mod = asciitable.read('fhd93521.dat')
w = mod['col1']
f = mod['col2']


def make_plots():
    for i in range(51):
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111)
        wl, fl = GW[i]
        ind = (w > (wl[0] - 2)) & (w < (wl[-1] + 2))
        ax.plot(w[ind], f[ind])
        ax.set_title("Order %s" % (i + 1))

        ax.xaxis.set_major_formatter(FSF("%.0f"))
        ax.set_xlabel(r"$\lambda\quad[\AA]$")
        ax.set_ylabel(r"$F_\lambda\quad[\frac{{\rm ergs}}{{\rm cm}^2 /{\rm s}/\AA}]$")
        fig.savefig("hd93521/{:0>2.0f}.png".format(i + 1))


def trim_kurucz():
    data = asciitable.read("vega_50000.dat")
    nm = data['col1']
    fl = data['col2']
    ind = (nm > 300) & (nm < 950)
    asciitable.write({"w": 10. * nm[ind], "fl": fl[ind]}, "vega_cut.dat", names=["w", "fl"])


def vega_bp():
    data = asciitable.read("mhr7001.dat")
    wl = data['col1']
    ab = data['col2']
    #ab = -2.5 * np.log10((wl/10.)**2/c_ang * fl) - 13
    bpass = np.empty((len(wl),))
    bpass[:-1] = np.diff(wl)
    bpass[-1] = bpass[-2]
    asciitable.write({"w": wl, "ab": ab, "bp": bpass}, "hr7001bp.dat", names=["w", "ab", "bp"])


def plot_vega():
    data = asciitable.read("vega_cut.dat")
    wl = data['w']
    fl = data['fl']
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    ax.plot(wl, fl)
    plt.show()


def main():
    #trim_kurucz()
    #plot_vega()
    #vega_bp()
    make_plots()


if __name__ == "__main__":
    main()
