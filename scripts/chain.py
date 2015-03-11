#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Measure statistics across multiple chains.")
parser.add_argument("--glob", help="Do something on this glob. Must be given as a quoted expression to avoid shell expansion.")

# parser.add_argument("--dir", action="store_true", help="Concatenate all of the flatchains stored
# within run* folders in the current directory. Designed to collate runs from a JobArray.")
parser.add_argument("--outdir", default="mcmcplot", help="Output directory to contain all plots.")
parser.add_argument("--output", default="combined.hdf5", help="Output HDF5 file.")

parser.add_argument("--files", nargs="+", help="The HDF5 or CSV files containing the MCMC samples, separated by whitespace.")

parser.add_argument("--burn", type=int, default=0, help="How many samples to discard from the beginning of the chain "
                                                        "for burn in.")
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
import numpy as np
import h5py
from astropy.table import Table
from astropy.io import ascii
import sys
import csv

#Check to see if outdir exists.
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

args.outdir += "/"



def h5read(fname, burn=0, thin=1):
    '''
    Read the flatchain from the HDF5 file and return it.
    '''
    fid = h5py.File(fname, "r")
    assert burn < fid["samples"].shape[0]
    print("{} burning by {} and thinning by {}".format(fname, burn, thin))
    flatchain = fid["samples"][burn::thin]

    fid.close()

    return flatchain

def csvread(fname, burn=0, thin=1):
    '''
    Read the flatchain from a CSV file and return it.
    '''
    flatchain = np.genfromtxt(fname, skip_header=1, dtype=float, delimiter=",")[burn::thin]

    return flatchain

def gelman_rubin(samplelist):
    '''
    Given a list of flatchains from separate runs (that already have burn in cut
    and have been trimmed, if desired), compute the Gelman-Rubin statistics in
    Bayesian Data Analysis 3, pg 284. If you want to compute this for fewer
    parameters, then slice the list before feeding it in.
    '''

    full_iterations = len(samplelist[0])
    assert full_iterations % 2 == 0, "Number of iterations must be even. Try cutting off a different number of burn in samples."
    shape = samplelist[0].shape
    #make sure all the chains have the same number of iterations
    for flatchain in samplelist:
        assert len(flatchain) == full_iterations, "Not all chains have the same number of iterations!"
        assert flatchain.shape == shape, "Not all flatchains have the same shape!"

    #make sure all chains have the same number of parameters.

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
    avg_phi_j = np.mean(chains, axis=0, dtype="f8") #average over iterations, now a (m, nparams) array
    #average value of all chains
    avg_phi = np.mean(chains, axis=(0,1), dtype="f8") #average over iterations and chains, now a (nparams,) array

    B = n/(m - 1.0) * np.sum((avg_phi_j - avg_phi)**2, axis=0, dtype="f8") #now a (nparams,) array

    s2j = 1./(n - 1.) * np.sum((chains - avg_phi_j)**2, axis=0, dtype="f8") #now a (m, nparams) array

    W = 1./m * np.sum(s2j, axis=0, dtype="f8") #now a (nparams,) arary

    var_hat = (n - 1.)/n * W + B/n #still a (nparams,) array
    std_hat = np.sqrt(var_hat)

    R_hat = np.sqrt(var_hat/W) #still a (nparams,) array


    data = Table({   "Value": avg_phi,
                     "Uncertainty": std_hat},
                 names=["Value", "Uncertainty"])

    print(data)

    ascii.write(data, sys.stdout, Writer = ascii.Latex, formats={"Value":"%0.3f", "Uncertainty":"%0.3f"}) #

    #print("Average parameter value: {}".format(avg_phi))
    #print("std_hat: {}".format(np.sqrt(var_hat)))
    print("R_hat: {}".format(R_hat))

    if np.any(R_hat >= 1.1):
        print("You might consider running the chain for longer. Not all R_hats are less than 1.1.")


def plot(flatchain, base=args.outdir, format=".png"):
    '''
    Make a triangle plot
    '''

    import triangle

    labels = [r"$M_\ast\quad [M_\odot]$", r"$r_c$ [AU]", r"$T_{10}$ [K]",
    r"$q$", r"$\log M_\textrm{CO} \quad \log [M_\oplus]$",  r"$\xi$ [km/s]",
    r"$d$ [pc]",
    r"$i_d \quad [{}^\circ]$", r"PA $[{}^\circ]$", r"$v_r$ [km/s]",
    r"$\mu_\alpha$ ['']", r"$\mu_\delta$ ['']"]
    figure = triangle.corner(flatchain, quantiles=[0.16, 0.5, 0.84],
        plot_contours=True, plot_datapoints=False, labels=labels, show_titles=True)
    figure.savefig(base + "triangle" + format)

def paper_plot(flatchain, base=args.outdir, format=".pdf"):
    '''
    Make a triangle plot of just M vs i
    '''

    import matplotlib
    matplotlib.rc("font", size=8)
    matplotlib.rc("lines", linewidth=0.5)
    matplotlib.rc("axes", linewidth=0.8)
    matplotlib.rc("patch", linewidth=0.7)
    import matplotlib.pyplot as plt
    #matplotlib.rc("axes", labelpad=10)
    from matplotlib.ticker import FormatStrFormatter as FSF
    from matplotlib.ticker import MaxNLocator
    import triangle

    labels = [r"$M_\ast\enskip [M_\odot]$", r"$i_d \enskip [{}^\circ]$"]
    #r"$r_c$ [AU]", r"$T_{10}$ [K]", r"$q$", r"$\log M_\textrm{CO} \enskip [\log M_\oplus]$",
    #r"$\xi$ [km/s]"]
    inds = np.array([0, 6, ]) #1, 2, 3, 4, 5])

    K = len(labels)
    fig, axes = plt.subplots(K, K, figsize=(3., 2.5))

    figure = triangle.corner(flatchain[:, inds], plot_contours=True,
    plot_datapoints=False, labels=labels, show_titles=False,
        fig=fig)

    for ax in axes[:, 0]:
        ax.yaxis.set_label_coords(-0.4, 0.5)
    for ax in axes[-1, :]:
        ax.xaxis.set_label_coords(0.5, -0.4)

    figure.subplots_adjust(left=0.2, right=0.8, top=0.95, bottom=0.2)

    figure.savefig(base + "ptriangle" + format)


def plot_walkers(flatchain, base=args.outdir, start=0, end=-1, labels=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    # majorLocator = MaxNLocator(nbins=4)
    ndim = len(flatchain[0, :])
    sample_num = np.arange(len(flatchain[:,0]))
    sample_num = sample_num[start:end]
    samples = flatchain[start:end]
    plt.rc("ytick", labelsize="x-small")

    # if lnprobs is not None:
    #     fig, ax = plt.subplots(nrows=ndim + 1, sharex=True)
    #     ax[0].plot(sample_num, lnprobs[start:end])
    #     ax[0].set_ylabel("lnprob")
    #     for i in range(0, ndim):
    #         ax[i+1].plot(sample_num, samples[:,i])
    #         ax[i+1].yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    #         if labels is not None:
    #             ax[i+1].set_ylabel(labels[i])


    fig, ax = plt.subplots(nrows=ndim, sharex=True)
    for i in range(0, ndim):
        ax[i].plot(sample_num, samples[:,i])
        ax[i].yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        if labels is not None:
            ax[i].set_ylabel(labels[i])

    ax[-1].set_xlabel("Sample number")
    fig.subplots_adjust(hspace=0)
    fig.savefig(base + "walkers.png")
    plt.close(fig)

def estimate_covariance(flatchain):

    if args.ndim:
        d = args.ndim
    else:
        d = flatchain.shape[1]

    import matplotlib.pyplot as plt

    #print("Parameters {}".format(flatchain.param_tuple))
    #samples = flatchain.samples
    cov = np.cov(flatchain, rowvar=0)

    #Now try correlation coefficient
    cor = np.corrcoef(flatchain, rowvar=0)
    print("Correlation coefficient")
    print(cor)

    # Make a plot of correlation coefficient.

    fig, ax = plt.subplots(figsize=(0.5 * d, 0.5 * d), nrows=1, ncols=1)
    ext = (0.5, d + 0.5, 0.5, d + 0.5)
    ax.imshow(cor, origin="upper", vmin=-1, vmax=1, cmap="bwr", interpolation="none", extent=ext)
    fig.savefig("cor_coefficient.png")

    print("'Optimal' jumps with covariance (units squared)")

    opt_jump = 2.38**2/d * cov
    # opt_jump = 1.7**2/d * cov # gives about ??
    print(opt_jump)

    print("Standard deviation")
    std_dev = np.sqrt(np.diag(cov))
    print(std_dev)

    print("'Optimal' jumps")
    if args.ndim:
        d = args.ndim
    else:
        d = flatchain.shape[1]
    print(2.38/np.sqrt(d) * std_dev)

    np.save("mcmcplot/opt_jump.npy", opt_jump)



def cat_list(file, flatchainList):
    '''
    Given a list of flatchains, concatenate all of these and write them to a
    single HDF5 file.
    '''
    #Write this out to the new file
    print("Opening", file)
    hdf5 = h5py.File(file, "w")

    cat = np.concatenate(flatchainList, axis=0)

    # id = flatchainList[0].id
    # param_tuple = flatchainList[0].param_tuple

    dset = hdf5.create_dataset("samples", cat.shape, compression='gzip',
        compression_opts=9)
    dset[:] = cat
    # dset.attrs["parameters"] = "{}".format(param_tuple)

    hdf5.close()

#Now that all of the structures have been declared, do the initialization stuff.

if args.glob:
    from glob import glob
    files = glob(args.glob)
elif args.files:
    files = args.files
else:
    import sys
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
            flatchainList.append(h5read(file, args.burn, args.thin))
        elif ext == ".csv":
            flatchainList.append(csvread(file, args.burn, args.thin))
    except OSError as e:
        print("{} does not exist, skipping. Or error {}".format(file, e))

print("Using a total of {} flatchains".format(len(flatchainList)))

if args.cat:
    assert len(flatchainList) > 1, "If concatenating samples, must provide more than one flatchain"
    cat_list(args.output, flatchainList)

# Assume that if we are using either args.chain, or args.triangle, we are only suppling one
# chain, since these commands don't make sense otherwise.
if args.chain:
    assert len(flatchainList) == 1, "If plotting Markov Chain, only specify one flatchain"
    plot_walkers(flatchainList[0])

if args.triangle:
    assert len(flatchainList) == 1, "If making Triangle, only specify one flatchain"
    plot(flatchainList[0])

if args.paper:
    assert len(flatchainList) == 1, "If making Triangle, only specify one flatchain"
    paper_plot(flatchainList[0])

if args.cov:
    assert len(flatchainList) == 1, "If estimating covariance, only specify one flatchain"
    estimate_covariance(flatchainList[0])

if args.gelman:
    assert len(flatchainList) > 1, "If running Gelman-Rubin test, must provide more than one flatchain"
    gelman_rubin(flatchainList)
