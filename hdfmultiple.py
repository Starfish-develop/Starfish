#!/usr/bin/env python
import os
import numpy as np

'''
hdfmultiple.py

Tools for measuring the output of many runs.

Main functionality:

1) Visualize the chains from all of the runs on the same plot. This way we can check convergence.
   1.1) Ability to plot the starting positions of each chain, stellar[0], as well.
2) Measure the between-chain correlation. Compute the Gelman-Rubin statistics.
'''

#Plot kw
label_dict = {"temp":r"$T_{\rm eff}$", "logg":r"$\log_{10} g$", "Z":r"$[{\rm Fe}/{\rm H}]$", "alpha":r"$[\alpha/{\rm Fe}]$",
              "vsini":r"$v \sin i$", "vz":r"$v_z$", "logOmega":r"$\log_{10} \Omega$", "logc0":r"$\log_{10} c_0$",
              "sigAmp":r"$b$", "logAmp":r"$\log_{10} a_{\rm g}", "l":r"$l$",
              "h":r"$h$", "loga":r"$\log_{10} a$", "mu":r"$\mu$", "sigma":r"$\sigma$"}

import argparse
parser = argparse.ArgumentParser(description="Measure statistics across multiple chains.")
parser.add_argument("--dir", action="store_true", help="Concatenate all of the flatchains stored within run* "
                                                       "folders in the current directory. Designed to collate runs from a JobArray.")
parser.add_argument("-o", "--outdir", default="mcmcplot", help="Output directory to contain all plots.")
parser.add_argument("--files", nargs="+", help="The HDF5 files containing the MCMC samples, separated by whitespace.")
parser.add_argument("--clobber", action="store_true", help="Overwrite existing files?")
parser.add_argument("--chain", action="store_true", help="Make a plot of the position of the chains.")
parser.add_argument("--burn", type=int, default=0, help="How many samples to discard from the beginning of the chain "
                                                        "for burn in.")
parser.add_argument("--thin", type=int, default=1, help="Thin the chain by this factor. E.g., --thin 100 will take "
                                                        "every 100th sample.")
parser.add_argument("--stellar_params", nargs="*", default="all", help="A list of which stellar parameters to plot, "
                                                                       "separated by WHITESPACE. Default is to plot all.")
parser.add_argument("--gelman", action="store_true", help="Compute the Gelman-Rubin convergence statistics.")

args = parser.parse_args()

#Check to see if outdir exists. If --clobber, overwrite, otherwise exit.
if os.path.exists(args.outdir):
    if not args.clobber:
        import sys
        sys.exit("Error: --outdir already exists and --clobber is not set. Exiting.")
else:
    os.makedirs(args.outdir)

args.outdir += "/"

if args.dir:
    #assemble all of the flatchains.hdf5 files from the run* subdirectories.
    import glob
    folders = glob.glob("run*")
    files = [folder + "/flatchains.hdf5" for folder in folders]
elif args.files:
    assert len(args.files) >= 2, "Must provide 2 or more HDF5 files to combine."
    files = args.files
else:
    import sys
    sys.exit("Must specify either --dir or --files")

import h5py
hdf5list = [h5py.File(file, "r") for file in files]
stellarlist = [hdf5.get("stellar") for hdf5 in hdf5list]

for stellar in stellarlist:
    assert stellar.attrs["parameters"] == stellarlist[0].attrs["parameters"], "Parameter lists do not match."
    assert stellar.shape[1] == stellarlist[0].shape[1], "Different number of parameters."

stellar_tuple = stellar.attrs["parameters"]
stellar_tuple = tuple([param.strip("'() ") for param in stellar_tuple.split(",")])

def find_cov(name):
    if "cov" in name:
        return True

def find_region(name):
    if "cov_region" in name:
        return True

#Determine how many orders, if there is global covariance, or regions
#choose the first chain
hdf5 = hdf5list[0]
orders = [int(key) for key in hdf5.keys() if key != "stellar"]
orders.sort()

yes_cov = hdf5.visit(find_cov)
print("Is there covariance: {}".format(yes_cov))
yes_region = hdf5.visit(find_region)

# give this a key relative from the top, and it will return a list of all flatchains
def get_flatchains(key):
    return [hdf5.get(key)[:] for hdf5 in hdf5list]

#Order list will always be a 2D list, with the items being flatchains
ordersList = []
for order in orders:

    print("Adding cheb for order {}".format(order))
    temp = [get_flatchains("{}/cheb".format(order))]
    if yes_cov:
        print("Adding cov for order {}".format(order))
        temp += [get_flatchains("{}/cov".format(order))]

    #TODO: do something about regions here

    #accumulate all of the orders
    ordersList += [temp]

# order22list = [order22cheblist, order22covlist]
# order23list = [order23cheblist, order23covlist]
# orderlist = [order22list, order23list]
# That way, we can do something like gelman_rubin(order22cheblist)

print("Thinning by ", args.thin)
print("Burning out first {} samples".format(args.burn))
stellarlist = [stellar[args.burn::args.thin] for stellar in stellarlist]
#a triple list comprehension is bad for readability, but I can't think of something better
ordersList = [[[flatchain[args.burn::args.thin] for flatchain in subList] for subList in orderList] for orderList in
              ordersList]

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
    stellarlist = [stellar[:, index_arr] for stellar in stellarlist]

stellar_labels = [label_dict[key] for key in stellar_params]

def gelman_rubin(samplelist):
    '''
    Given a list of flatchains from separate runs (that already have burn in cut and have been trimmed, if desired),
    compute the Gelman-Rubin statistics in Bayesian Data Analysis 3, pg 284.

    If you want to compute this for fewer parameters, then truncate the list before feeding it in.
    '''

    full_iterations = len(samplelist[0])
    assert full_iterations % 2 == 0, "Number of iterations must be even. Try cutting off a different number of burn " \
                                     "in samples."
    #make sure all the chains have the same number of iterations
    for flatchain in samplelist:
        assert len(flatchain) == full_iterations, "Not all chains have the same number of iterations!"

    #Following Gelman,
    # n = length of split chains
    # i = index of iteration in chain
    # m = number of split chains
    # j = index of which chain
    n = full_iterations//2
    m = 2 * len(samplelist)
    nparams = samplelist[0].shape[-1] #the trailing dimension of a flatchain

    #Block the chains up into a 3D array
    chains = np.empty((n, m, nparams))
    for k, flatchain in enumerate(samplelist):
        chains[:,2*k,:] = flatchain[:n]  #first half of chain
        chains[:,2*k + 1,:] = flatchain[n:] #second half of chain

    #Now compute statistics
    #average value of each chain
    avg_phi_j = np.average(chains, axis=0) #average over iterations, now a (m, nparams) array
    #average value of all chains
    avg_phi = np.average(chains, axis=(0,1)) #average over iterations and chains, now a (nparams,) array

    B = n/(m - 1.0) * np.sum((avg_phi_j - avg_phi)**2, axis=0) #now a (nparams,) array

    s2j = 1./(n - 1.) * np.sum((chains - avg_phi_j)**2, axis=0) #now a (m, nparams) array

    W = 1./m * np.sum(s2j, axis=0) #now a (nparams,) arary

    var_hat = (n - 1.)/n * W + B/n #still a (nparams,) array

    R_hat = np.sqrt(var_hat/W) #still a (nparams,) array

    print("Between-sequence variance B: {}".format(B))
    print("Within-sequence variance W: {}".format(W))
    print("var_hat: {}".format(var_hat))
    print("std_hat: {}".format(np.sqrt(var_hat)))
    print("R_hat: {}".format(R_hat))

    if np.any(R_hat >= 1.1):
        print("You might consider running the chain for longer. Not all R_hats are less than 1.1.")


if args.gelman:
    #Compute the Gelman-Rubin statistics BDA 3, pg 284
    print("Stellar parameters")
    gelman_rubin(stellarlist)
    for i, orderList in enumerate(ordersList):
        print("\nOrder {}".format(orders[i]))
        for subList in orderList:
            gelman_rubin(subList)



[hdf5.close for hdf5 in hdf5list]
