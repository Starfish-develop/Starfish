'''
Combine the inference of many HDF5 regions.

Works by matching regions within some degree of variance.
'''

import os
import numpy as np

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


#For each order within each flatchains.hdf5, come up with a list of region means

#How should we pair regions?

#First, we need to come up with a list of all regions, perhaps a {key:(run23/flatchains.hdf/22/cov_region02),
# mu: 5129, range: 2}, where range is the number of angstroms that we believe could possibly overlap with this region
# . Range could be determined either by a fixed number, the variance of the mean, or the width of the region itself.

#Next, given this list of dictionaries, sort it into sets of keys that are mutually consistent with each other.
# Does this mean that we should be choosing a unified range based upon all regions, and finding those which fall into
#  it? Or should we choose one region and then see what else matches it?

# This seems like a tricky job for a classifier

#Specify an amplitude cut. i.e., if the mean amplitude (after some specified period of front-burn in) is below,
# then it is treated as though this region did not exist.

#Try this
#1) start with the first flatchain in the list
#2) Create N_region dictionaries, each one {mu: XX, var: XX, keylist: [], mu_list: [], var_list: []}
#3) for the 2nd flatchain, loop through the regions. See if they fit into anything by checking if abs(mu1 - mu0) is
# less than var. If so, add it to the list and then recompute mu, var.
#4) if the region does not fit into any of the previously existing lists of regions, then create a new list of
# regions (print that you are doing this)
#5) standardize the lengths of chains by specifying complement of burn in, i.e., counting from the back of the chain
# how many samples do we want to keep.
#6) Thin the chains if desired
#6) At the end, compute the GR diagnostic for all of these lists of chains.
#7) Then create a combined.hdf5 that contains all of the sampled regions.


# This might be a great opportunity to create a Flatchains() Python object. which automatically has add capability.
# For example, hdfcat could just be flatchains + flatchains. Also there could be burn-in methods, plot methods,
# etc. But I don't know if it's really worth it.