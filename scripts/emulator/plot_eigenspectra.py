'''
After running `create_PCA.py`, use this script to examine the output.
'''

import argparse
parser = argparse.ArgumentParser(prog="plot_eigenspectra.py", description="Determine how well the eigenspectra "
                                                                          "reproduce the grid.")
parser.add_argument("input", help="*.yaml file specifying parameters.")
args = parser.parse_args()

import yaml

f = open(args.input)
cfg = yaml.load(f)
f.close()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
from Starfish.grid_tools import HDF5Interface
from Starfish.emulator import PCAGrid

pcagrid = PCAGrid.from_cfg(cfg)
ind = pcagrid.ind

#Make sure that we can get the same indices from the main grid.
grid = HDF5Interface(cfg["grid"], ranges=cfg["ranges"])
wl = grid.wl[ind]

temps = np.unique(pcagrid.gparams[:,0])
loggs = np.unique(pcagrid.gparams[:,1])
Zs = np.unique(pcagrid.gparams[:,2])
points = {"temp":temps, "logg":loggs, "Z":Zs}

base = cfg['outdir']
# Plot the eigenspectra

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)

ax.xaxis.set_major_formatter(FSF("%.0f"))
ax.set_xlabel(r"$\lambda$ [\AA]")
ax.set_ylabel(r"$\propto f_\lambda$")
    
ax.set_title("Eigenspectra")

for i,comp in enumerate(pcagrid.pcomps):
    ax.plot(wl, comp, label="{}".format(i+1))
    
ax.legend(loc="lower left")
fig.savefig(base + "eigenspectra.png")

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.xaxis.set_major_formatter(FSF("%.0f"))
ax.set_xlabel(r"$\lambda$ [\AA]")
ax.set_ylabel(r"$\propto f_\lambda$")
ax.plot(wl, pcagrid.flux_mean)
ax.set_title("Mean flux")
fig.savefig(base + "mean.png")


fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.xaxis.set_major_formatter(FSF("%.0f"))
ax.set_xlabel(r"$\lambda$ [\AA]")
ax.set_ylabel(r"$\propto f_\lambda$")
ax.plot(wl, pcagrid.flux_std)
ax.set_title("Std flux")
fig.savefig(base + "std.png")
plt.clf()

# Reconstruct the original grid and see what the total error is.

recon_fluxes = pcagrid.reconstruct_all()
fluxes = np.array([flux[ind] for flux in grid.fluxes])

#Normalize all of the fluxes to an average value of 1
#In order to remove interesting correlations

fluxes = fluxes/np.average(fluxes, axis=1)[np.newaxis].T

frac_err = (fluxes - recon_fluxes)/fluxes

print("Max fractional error {:.2f}%".format(100*np.max(np.abs(frac_err))))
print("Std fractional error {:.2f}%".format(100*np.std(frac_err)))

fig, ax = plt.subplots(nrows=2, figsize=(20, 5), sharex=True)
ax[0].plot(fluxes.flatten())
ax[0].plot(recon_fluxes.flatten())
ax[1].plot(frac_err[0,:] * 100.)
ax[1].set_ylabel("Error [\%]")
fig.savefig(base + "reconstruct.png")
plt.show()
