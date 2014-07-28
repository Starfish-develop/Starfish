#!/usr/bin/env python
import os
import numpy as np


'''
Script designed to plot the HDF5 output from MCMC runs.
'''

#Plot kw
label_dict = {"temp":r"$T_{\rm eff}$", "logg":r"$\log_{10} g$", "Z":r"$[{\rm Fe}/{\rm H}]$", "alpha":r"$[\alpha/{\rm Fe}]$",
    "vsini":r"$v \sin i$", "vz":r"$v_z$", "logOmega":r"$\log_{10} \Omega$", "logc0":r"$\log_{10} c_0$",
    "c1":r"$c_0$", "c2":r"$c_1$", "c3":r"$c_3$",
    "sigAmp":r"$b$", "logAmp":r"$\log_{10} a_{\rm g}", "l":r"$l$",
    "h":r"$h$", "loga":r"$\log_{10} a$", "mu":r"$\mu$", "sigma":r"$\sigma$"}


import argparse
parser = argparse.ArgumentParser(description="Plot the MCMC output. Default is to plot everything possible.")
parser.add_argument("HDF5file", help="The HDF5 file containing the MCMC samples.")
parser.add_argument("-o", "--outdir", default="mcmcplot", help="Output directory to contain all plots.")
parser.add_argument("--clobber", action="store_true", help="Overwrite existing output directory?")

parser.add_argument("-t", "--triangle", action="store_true", help="Make a triangle (staircase) plot of the parameters.")
parser.add_argument("--chain", action="store_true", help="Make a plot of the position of the chains.")
parser.add_argument("--acor", action="store_true", help="Calculate the autocorrelation of the chain")
parser.add_argument("--acor-window", type=int, default=50, help="window to compute acor with")

parser.add_argument("--cov", action="store_true", help="Estimate the covariance between two parameters.")

parser.add_argument("--burn", type=int, default=0, help="How many samples to discard from the beginning of the chain "
                                                        "for burn in.")
parser.add_argument("--thin", type=int, default=1, help="Thin the chain by this factor. E.g., --thin 100 will take "
                                                        "every 100th sample.")
parser.add_argument("--stellar_params", nargs="*", default="all", help="A list of which stellar parameters to plot, "
                                                                    "separated by WHITESPACE. Default is to plot all.")
args = parser.parse_args()

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

print("Thinning by ", args.thin)
print("Burning out first {} samples".format(args.burn))
stellar = stellar[args.burn::args.thin]

if args.stellar_params == "all":
    stellar_params = stellar_tuple
else:
    stellar_params = args.stellar_params
    #Figure out which rows we need to select
    index_arr = []
    for param in stellar_params:
        #What index is this param in stellar_tuple?
        index_arr += [stellar_tuple.index(param)]
    index_arr = np.array(index_arr)
    stellar = stellar[:, index_arr]

stellar_labels = [label_dict[key] for key in stellar_params]

def find_cov(name):
    if "cov" in name:
        return True
    return None

def find_region(name):
    if "cov_region" in name:
        return True
    return None

#Determine how many orders, if there is global covariance, or regions
#choose the first chain
orders = [int(key) for key in hdf5.keys() if key != "stellar"]
orders.sort()

yes_cov = hdf5.visit(find_cov)
yes_region = hdf5.visit(find_region)
#Order list will always be a 2D list, with the items being flatchains
cheb_parameters = hdf5.get("{}/cheb".format(orders[0])).attrs["parameters"]
cheb_tuple = tuple([param.strip("'() ") for param in cheb_parameters.split(",")])
cheb_labels = [label_dict[key] for key in cheb_tuple]
if yes_cov:
    cov_parameters = hdf5.get("{}/cov".format(orders[0])).attrs["parameters"]
    cov_tuple = tuple([param.strip("'() ") for param in cov_parameters.split(",")])
    cov_labels = [label_dict[key] for key in cov_tuple]

ordersList = []
for order in orders:

    temp = [hdf5.get("{}/cheb".format(order))]
    if yes_cov:
        temp += [hdf5.get("{}/cov".format(order))]

    #TODO: do something about regions here
    #accumulate all of the orders
    ordersList += [temp]

if args.triangle:
    import matplotlib
    matplotlib.rc("axes", labelsize="large")
    import triangle
    figure = triangle.corner(stellar, labels=stellar_labels, quantiles=[0.16, 0.5, 0.84],
                             show_titles=True, title_args={"fontsize": 12})
    figure.savefig(args.outdir + "stellar_triangle.png")

    #Now plot all the other parameters
    for i, order in enumerate(orders):
        orderList = ordersList[i]
        cheb = orderList[0]
        figure = triangle.corner(cheb, labels=cheb_labels, quantiles=[0.16, 0.5, 0.84],
                                 show_titles=True, title_args={"fontsize": 12})
        figure.savefig(args.outdir + "{}_cheb_triangle.png".format(order))

        if yes_cov:
            cov = orderList[1]
            figure = triangle.corner(cov, labels=cov_labels, quantiles=[0.16, 0.5, 0.84],
                                     show_titles=True, title_args={"fontsize": 12})
            figure.savefig(args.outdir + "{}_cov_triangle.png".format(order))


if args.cov:
    '''
    Estimate the covariance of the remaining samples.
    '''
    print("Estimating covariances")
    print(stellar_params)

    cov = np.cov(stellar, rowvar=0)

    print("Standard deviation")
    print(np.sqrt(np.diag(cov)))

    print("Covariance")
    print(cov)

    #Now try correlation coefficient
    cor = np.corrcoef(stellar, rowvar=0)
    print("Correlation coefficient")
    print(cor)

# plot_walkers(self.outdir + self.fname + "_chain_pos.png", samples, labels=self.param_tuple)
# plt.close(figure)

if args.acor:
    from emcee import autocorr
    print("Stellar Autocorrelation")
    acors = autocorr.integrated_time(stellar, axis=0, window=args.acor_window)
    for param,acor in zip(stellar_tuple, acors):
        print("{} : {}".format(param, acor))






hdf5.close()