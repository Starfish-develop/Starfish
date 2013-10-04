import matplotlib.pyplot as plt
import numpy as np
import pyfits as pf

from matplotlib.ticker import FormatStrFormatter as FSF
import asciitable


def load_sens(filename):
    sfunc, hdr = pf.getdata(filename, header=True)
    wl = hdr['CDELT1'] * np.arange(len(sfunc)) + hdr['CRVAL1']
    return [wl, sfunc]


def plot_sens():
    wl, sfunc = load_sens('Feige34/sens_feige34.0025.fits')
    plt.plot(wl, sfunc)
    plt.show()


bfile1 = "Vega/sens_vega.00" #.fits
#bfile2 = "Feige34/sens_feige34.00"
bfile3 = "HD93521/sens_hd93521.00"


def plot_all_sensfunc():
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(11, 8))
    for i in range(5):
        for j in range(5):
            order = i * 5 + j + 25

            wl, sfunc = load_sens(bfile1 + "{:0>2.0f}.fits".format(order + 1))
            ax[i, j].plot(wl, sfunc, "b")

            #wl2,sfunc2 = load_sens(bfile2 + "{:0>2.0f}.fits".format(order+1))
            #ax[i,j].plot(wl2,sfunc2,"g")

            wl3, sfunc3 = load_sens(bfile3 + "{:0>2.0f}.fits".format(order + 1))
            ax[i, j].plot(wl3, sfunc3 - .8, "r")

            ax[i, j].xaxis.set_major_formatter(FSF("%.0f"))
            ax[i, j].locator_params(axis='x', nbins=5)
            ax[i, j].set_xlim(wl[0], wl[-1])
            ax[i, j].annotate("%s" % (order + 1), (0.1, 0.85), xycoords="axes fraction")
    fig.subplots_adjust(left=0.04, bottom=0.04, top=0.97, right=0.97)
    ax[4, 0].set_xlabel(r"$\lambda\quad[\AA]$")
    ax[-1, -1].set_ylim(34, 35.5)
    fig.savefig('sensfunc_50.png')


def main():
    plot_all_sensfunc()


if __name__ == "__main__":
    main()
