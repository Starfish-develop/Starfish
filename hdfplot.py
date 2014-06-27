#!/usr/bin/env python
import os
import numpy as np


'''
Script designed to plot the HDF5 output from MCMC runs.
'''

#Plot kw
label_dict = {"temp":r"$T_{\rm eff}$", "logg":r"$\log_{10} g$", "Z":r"$[{\rm Fe}/{\rm H}]$", "alpha":r"$[\alpha/{\rm Fe}]$",
    "vsini":r"$v \sin i$", "vz":r"$v_z$", "logOmega":r"$\log_{10} \Omega$", "logc0":r"$\log_{10} c_0$",
    "sigAmp":r"$b$", "logAmp":r"$\log_{10} a_{\rm g}", "l":r"$l$",
    "h":r"$h$", "loga":r"$\log_{10} a$", "mu":r"$\mu$", "sigma":r"$\sigma$"}


import argparse
parser = argparse.ArgumentParser(description="Plot the MCMC output. Default is to plot everything possible.")
parser.add_argument("HDF5file", help="The HDF5 file containing the MCMC samples.")
parser.add_argument("-o", "--outdir", default="mcmcplot", help="Output directory to contain all plots.")
parser.add_argument("--clobber", action="store_true", help="Overwrite existing output directory?")

parser.add_argument("-t", "--triangle", action="store_true", help="Make a triangle (staircase) plot of the parameters.")
parser.add_argument("--chain", action="store_true", help="Make a plot of the position of the chains.")

parser.add_argument("--stellar_params", nargs="*", default="all", help="A list of which stellar parameters to plot, "
                                                                    "separated by whitespace. Default is to plot all.")
args = parser.parse_args()

#Check for c1, c2, ...

#Check to see if outdir exists. If --clobber, overwrite, otherwise exit.
if os.path.exists(args.outdir):
    if not args.clobber:
        import sys
        sys.exit("Error: --outdir already exists and --clobber is not set. Exiting.")
else:
    os.makedirs(args.outdir)

args.outdir += "/"


#Load the HDF5 file
import h5py

hdf5 = h5py.File(args.HDF5file, "r")

#Load stellar samples and parameters
stellar = hdf5.get("stellar")
if stellar is None:
    sys.exit("HDF5 file contains no stellar samples.")

#Determine which parameters we want to plot from --stellar-params
stellar_tuple = stellar.attrs["parameters"]
stellar_tuple = tuple([param.strip("'() ") for param in stellar_tuple.split(",")])

print("Stellar tuple is" stellar_tuple)

if args.stellar_params == "all":
    stellar_params = stellar_tuple
else:
    stellar_params = args.stellar_params
    #Figure out which rows we need to select
    index_arr = []
    for param in stellar_params:
        #What index is this param in stellar_tuple?
        print("Checking param", param)
        index_arr += [stellar_tuple.index(param)]
    index_arr = np.array(index_arr)
    stellar = stellar[:, index_arr]

stellar_labels = [label_dict[key] for key in stellar_params]

if args.triangle:
    import matplotlib
    matplotlib.rc("axes", labelsize="large")
    import triangle
    figure = triangle.corner(stellar, labels=stellar_labels, quantiles=[0.16, 0.5, 0.84],
                             show_titles=True, title_args={"fontsize": 12})
    figure.savefig(args.outdir + "stellar_triangle.png")



# plot_walkers(self.outdir + self.fname + "_chain_pos.png", samples, labels=self.param_tuple)
# plt.close(figure)


orderList = [int(key) for key in hdf5.keys() if key != "stellar"] #The remaining folders in the HDF5 file are order nums






#Determine how many orders there are

    #Determine how many regions there are




hdf5.close()