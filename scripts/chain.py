#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Measure statistics from MCMC runs, either for a single chain or across multiple chains.")
parser.add_argument("--glob", help="Do something on this glob. Must be given as a quoted expression to avoid shell expansion.")

parser.add_argument("--outdir", default="", help="Output directory to contain all plots.")
parser.add_argument("--output", default="combined.hdf5", help="Output HDF5 file.")

parser.add_argument("--files", nargs="+", help="The HDF5 or CSV files containing the MCMC samples, separated by whitespace.")

parser.add_argument("--burn", type=int, default=0, help="How many samples to discard from the beginning of the chain for burn in.")
parser.add_argument("--thin", type=int, default=1, help="Thin the chain by this factor. E.g., --thin 100 will take every 100th sample.")
parser.add_argument("--range", nargs=2, help="start and end ranges for chain plot, separated by WHITESPACE")

parser.add_argument("--cat", action="store_true", help="Concatenate the list of samples.")
parser.add_argument("--chain", action="store_true", help="Make a plot of the position of the chains.")
parser.add_argument("-t", "--triangle", action="store_true", help="Make a triangle (staircase) plot of the parameters.")
parser.add_argument("--params", nargs="*", default="all", help="A list of which parameters to plot,                                                                 separated by WHITESPACE. Default is to plot all.")

parser.add_argument("--gelman", action="store_true", help="Compute the Gelman-Rubin convergence statistics.")

parser.add_argument("--acor", action="store_true", help="Calculate the autocorrelation of the chain")
parser.add_argument("--acor-window", type=int, default=50, help="window to compute acor with")

parser.add_argument("--cov", action="store_true", help="Estimate the covariance between two parameters.")
parser.add_argument("--ndim", type=int, help="How many dimensions to use for estimating the 'optimal jump'.")
parser.add_argument("--paper", action="store_true", help="Change the figure plotting options appropriate for the paper.")

args = parser.parse_args()

import os
import sys

from Starfish import utils

#Check to see if outdir exists.
if not os.path.exists(args.outdir) and args.outdir != "":
    os.makedirs(args.outdir)
    args.outdir += "/"


# Now that all of the structures have been declared, do the initialization stuff.
# Wrap all of these calls inside of the main function, so that we can import other methods to do stuff

if args.glob:
    from glob import glob
    files = glob(args.glob)
elif args.files:
    files = args.files
else:
    sys.exit("Must specify either --glob or --files")

#Because we are impatient and want to compute statistics before all the jobs are finished,
# there may be some directories that do not have a flatchains.hdf5 file
flatchainList = []
for file in files:
    try:
        # If we've specified HDF5, use h5read
        # If we've specified csv, use csvread
        root, ext = os.path.splitext(file)
        if ext == ".hdf5":
            flatchainList.append(utils.h5read(file, args.burn, args.thin))
        elif ext == ".csv":
            flatchainList.append(utils.csvread(file, args.burn, args.thin))
    except OSError as e:
        print("{} does not exist, skipping. Or error {}".format(file, e))

print("Using a total of {} flatchains".format(len(flatchainList)))

if args.cat:
    assert len(flatchainList) > 1, "If concatenating samples, must provide more than one flatchain"
    utils.cat_list(args.output, flatchainList)

# Assume that if we are using either args.chain, or args.triangle, we are only suppling one
# chain, since these commands don't make sense otherwise.
if args.chain:
    assert len(flatchainList) == 1, "If plotting Markov Chain, only specify one flatchain"
    utils.plot_walkers(flatchainList[0], base=args.outdir)

if args.triangle:
    assert len(flatchainList) == 1, "If making Triangle, only specify one flatchain"
    utils.plot(flatchainList[0], base=args.outdir)

if args.paper:
    assert len(flatchainList) == 1, "If making Triangle, only specify one flatchain"
    utils.paper_plot(flatchainList[0], base=args.outdir)

if args.cov:
    assert len(flatchainList) == 1, "If estimating covariance, only specify one flatchain"
    utils.estimate_covariance(flatchainList[0], base=args.outdir)

if args.gelman:
    assert len(flatchainList) > 1, "If running Gelman-Rubin test, must provide more than one flatchain"
    utils.gelman_rubin(flatchainList)
