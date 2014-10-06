'''
Make a bunch of test draws from the GP to determine how well the process actually interpolates the grid.
'''

import argparse
parser = argparse.ArgumentParser(prog="plot_samples.py", description="Determine how well the Gaussian Process "
                                                                          "can interpolate the grid.")
parser.add_argument("input", help="*.yaml file specifying parameters.")
parser.add_argument("--index", type=int, default="-1", help="Which weight index to plot up. Default is to plot all.")
args = parser.parse_args()

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
from Starfish.emulator import PCAGrid, WeightEmulator

f = open(args.input)
cfg = yaml.load(f)
f.close()

#Load individual samples and then concatenate them
base = cfg["outdir"]
samples = np.array([np.load(base + "samples_w{}.npy".format(i)) for i in range(5)])


pcagrid = PCAGrid.from_cfg(cfg)

temps = np.unique(pcagrid.gparams[:,0])
loggs = np.unique(pcagrid.gparams[:,1])
Zs = np.unique(pcagrid.gparams[:,2])
points = {"temp":temps, "logg":loggs, "Z":Zs}
nts = len(temps)
nls = len(loggs)
nzs = len(Zs)

int_temps = np.linspace(temps[0], temps[-1], num=40)
int_loggs = np.linspace(loggs[0], loggs[-1], num=40)
int_Zs = np.linspace(Zs[0], Zs[-1], num=40)


def explore(weight_index):
    weights = pcagrid.w[weight_index]
    #Set up the emulator
    EMw = WeightEmulator(pcagrid, None, weight_index, samples[weight_index])

    nsamp = 5

    figt, axt = plt.subplots(nrows=4, ncols=4, figsize=(12,12), sharex=True, sharey=True)

    for i in range(nls):
        for j in range(nzs):
            logg = loggs[i]
            Z = Zs[j]
            ww = []
            for temp in temps:
                pars = np.array([temp, logg, Z])
                index = pcagrid.get_index(pars)
                ww.append(weights[index])
            axt[i,j].plot(temps, ww, "k")
            axt[i,j].plot(temps, ww, "bo")
            axt[i,j].annotate(r"$\log g = {:.1f}$".format(logg), (0.1, 0.90), xycoords="axes fraction", ha="left", color="k", size=9)
            axt[i,j].annotate(r"$Z = {:.1f}$".format(Z), (0.1, 0.8), xycoords="axes fraction", ha="left", color="k", size=9)

            fparams = []
            for temp in int_temps:
                fparams.append([temp, logg, Z])
            fparams = np.array(fparams)

            for k in range(nsamp):
                axt[i,j].plot(int_temps, EMw(fparams), "b", lw=0.5)

    axt[-1, -1].xaxis.set_major_formatter(FSF("%.0f"))
    axt[-1, -1].xaxis.set_major_locator(MultipleLocator(200))


    figl, axl = plt.subplots(nrows=4, ncols=7, figsize=(18,12), sharex=True, sharey=True)

    for i in range(nzs):
        for j in range(nts):
            Z = Zs[i]
            temp = temps[j]
            ww = []
            for logg in loggs:
                pars = np.array([temp, logg, Z])
                index = pcagrid.get_index(pars)
                ww.append(weights[index])
            axl[i,j].plot(loggs, ww, "k")
            axl[i,j].plot(loggs, ww, "bo")
            axl[i,j].annotate(r"$T_{\rm eff}$" + "$= {:.0f}$".format(temp), (0.1, 0.90), xycoords="axes fraction", ha="left", color="k", size=9)
            axl[i,j].annotate(r"$Z = {:.1f}$".format(Z), (0.1, 0.8), xycoords="axes fraction", ha="left", color="k", size=9)

            fparams = []
            for logg in int_loggs:
                fparams.append([temp, logg, Z])
            fparams = np.array(fparams)

            for k in range(nsamp):
                #EMw.emulator_params = samples[np.random.choice(indexes)]
                axl[i,j].plot(int_loggs, EMw(fparams), "b", lw=0.5)

    axl[-1, -1].xaxis.set_major_formatter(FSF("%.1f"))
    axl[-1, -1].xaxis.set_major_locator(MultipleLocator(0.5))

    figz, axz = plt.subplots(nrows=4, ncols=7, figsize=(18,12), sharex=True, sharey=True)

    for i in range(nls):
        for j in range(nts):
            logg = loggs[i]
            temp = temps[j]
            ww = []
            for Z in Zs:
                pars = np.array([temp, logg, Z])
                index = pcagrid.get_index(pars)
                ww.append(weights[index]) #np.sum(emulator.fluxes[index] * pcomp))
            axz[i,j].plot(Zs, ww, "k")
            axz[i,j].plot(Zs, ww, "bo")
            axz[i,j].annotate(r"$\log g = {:.1f}$".format(logg), (0.1, 0.90), xycoords="axes fraction", ha="left", color="k", size=9)
            axz[i,j].annotate(r"$T_{\rm eff}$" + "$ = {:.0f}$".format(temp), (0.1, 0.8), xycoords="axes fraction", ha="left", color="k", size=9)

            fparams = []
            for Z in int_Zs:
                fparams.append([temp, logg, Z])
            fparams = np.array(fparams)

            for k in range(nsamp):
                #EMw.emulator_params = samples[np.random.choice(indexes)]
                axz[i,j].plot(int_Zs, EMw(fparams), "b", lw=0.5)

    axz[-1, -1].xaxis.set_major_formatter(FSF("%.1f"))
    axz[-1, -1].xaxis.set_major_locator(MultipleLocator(0.5))


    figt.savefig(base + "weight{}_temp.png".format(weight_index))
    figl.savefig(base + "weight{}_logg.png".format(weight_index))
    figz.savefig(base + "weight{}_Z.png".format(weight_index))

ncomp = pcagrid.ncomp

if args.index < 0:
    for i in range(ncomp):
        explore(i)

else:
    assert args.index < ncomp, "There are only {} PCA components to choose from.".format(args.index)
    explore(args.index)
