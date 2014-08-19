#!/usr/bin/env python
'''
Combine the inference of many HDF5 regions.

Works by matching regions within some degree of variance.
'''

import os
import numpy as np
from collections import deque

import argparse
parser = argparse.ArgumentParser(description="Measure statistics across multiple chains.")
parser.add_argument("--dir", action="store_true", help="Concatenate all of the flatchains stored within run* "
                                                       "folders in the current directory. Designed to collate runs from a JobArray.")
parser.add_argument("-o", "--outdir", default="mcmcplot", help="Output directory to contain all plots.")
parser.add_argument("--files", nargs="+", help="The HDF5 files containing the MCMC samples, separated by whitespace.")
parser.add_argument("--clobber", action="store_true", help="Overwrite existing files?")
parser.add_argument("--chain", action="store_true", help="Make a plot of the position of the chains.")
parser.add_argument("--keep", type=int, default=0, help="How many samples to keep from the end of the chain, "
                                                        "the beginning of the chain will be for burn in.")
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
#Because we are impatient and want to compute statistics before all the jobs are finished, there may be some
# directories that do not have a flatchains.hdf5 file
hdf5list = []
filelist = []
for file in files:
    try:
        hdf5list += [h5py.File(file, "r")]
        filelist += [file]
    except OSError:
        print("{} does not exist, skipping.".format(file))

#I think we should separate this by order

#Load all of the samples from the a given order into one giant deque
allRegions = deque()

hdf5 = hdf5list[0]
orders = [int(key) for key in hdf5.keys() if key != "stellar"]
orders.sort()

#ordersList is a list of deques that contain the flatchains
ordersList = [deque() for order in orders]

for hdf5 in hdf5list:
    for i, order in enumerate(orders):
        deq = ordersList[i]
        #figure out list of which regions are in this order
        regionKeys = [key for key in hdf5["{}".format(order)].keys() if "cov_region" in key]
        for key in regionKeys:
            deq.append(hdf5.get("{}/{}".format(order, key))[:])

#Maybe before we try grouping all of the regions together, it would be better to simply plot the mean and variance of
#each region, that way we can see where everything lands?

mus = deque()
sigmas = deque()
for flatchain in ordersList[0]:
    samples = flatchain[:, 1]
    mu, sigma = np.mean(samples, dtype="f8"), np.std(samples, dtype="f8")
    mus.append(mu)
    sigmas.append(sigma)

import matplotlib.pyplot as plt
plt.errorbar(mus, np.arange(len(mus)), xerr=sigmas, fmt="o")
plt.savefig("regions.png")

#To keep track of what's being added to what
ID = 0

class Region:
    def __init__(self, flatchain=None):
        self.flatchains = [flatchain] if flatchain is not None else []
        #Assume that these flatchains are shape (Niterations, 3)
        self.params = ("loga", "mu", "sigma")
        global ID
        self.id = ID
        ID += 1
        print("Created region {} with mu={} +/- {}".format(self.id, self.mu, self.std))

    @property
    def mu(self):
        '''
        Compute the mean mu of all of the samples.
        '''
        return np.mean(np.concatenate([chain[:, 1] for chain in self.flatchains]), dtype="f8")

    @property
    def std(self):
        '''
        Compute the standard deviation of mu of all of the samples.
        '''
        return np.std(np.concatenate([chain[:, 1] for chain in self.flatchains]), dtype="f8")

    def check_and_append(self, flatchain):
        '''
        Given the current status of the region (namely, mu and var), should we add this flatchain? If so,
        add and return True to break out of a logic loop.
        '''
        #Compute mu of flatchain under consideration
        mu = np.mean(flatchain[:,1], dtype="f8")
        std = np.std(flatchain[:,1], dtype="f8")
        if np.abs(mu - self.mu) <= (3 * self.std):
            print("Current Region {}: mu={}+/-{}. Adding new flatchain with mu={}+/-{}".format(self.id, self.mu,
                                                                                self.std, mu, std))
            self.flatchains.append(flatchain)
            return True
        else:
            return False

    def keep(self, N=0, thin=1):
        '''
        Keep the last N samples from each chain, also thin.
        '''
        #First, find the shortest flatchain.
        shortest = np.min([len(flatchain) for flatchain in self.flatchains])
        if N > 0:
            assert N <= shortest, "Cannot keep more samples than the shortest flatchain."
        else:
            N = shortest
        print("Keeping last {} samples and thinning by {}".format(N, thin))
        self.flatchains = [flatchain[-N::thin] for flatchain in self.flatchains]

    def gelman_rubin(self):
        '''
        Given a list of flatchains from separate runs (that already have burn in cut and have been trimmed, if desired),
        compute the Gelman-Rubin statistics in Bayesian Data Analysis 3, pg 284.

        If you want to compute this for fewer parameters, then truncate the list before feeding it in.
        '''

        samplelist = self.flatchains
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
        avg_phi_j = np.mean(chains, axis=0, dtype="f8") #average over iterations, now a (m, nparams) array
        #average value of all chains
        avg_phi = np.mean(chains, axis=(0,1), dtype="f8") #average over iterations and chains, now a (nparams,) array
        print("Average parameter value: {}".format(avg_phi))

        B = n/(m - 1.0) * np.sum((avg_phi_j - avg_phi)**2, axis=0, dtype="f8") #now a (nparams,) array

        s2j = 1./(n - 1.) * np.sum((chains - avg_phi_j)**2, axis=0, dtype="f8") #now a (m, nparams) array

        W = 1./m * np.sum(s2j, axis=0, dtype="f8") #now a (nparams,) arary

        var_hat = (n - 1.)/n * W + B/n #still a (nparams,) array

        R_hat = np.sqrt(var_hat/W) #still a (nparams,) array

        print("Between-sequence variance B: {}".format(B))
        print("Within-sequence variance W: {}".format(W))
        print("std_hat: {}".format(np.sqrt(var_hat)))
        print("R_hat: {}".format(R_hat))

        if np.any(R_hat >= 1.1):
            print("You might consider running the chain for longer. Not all R_hats are less than 1.1.")

ordersRegions = [[] for order in orders] #2D list
for i,flatchain_deque in enumerate(ordersList):
    #Choose an order
    orderRegions = ordersRegions[i]
    #cycle through the deque and either add to current regions or create new ones
    while flatchain_deque:
        flatchain = flatchain_deque.pop()
        #See if this can be added to any of the existing regions
        added = False
        for region in orderRegions:
            added = region.check_and_append(flatchain)
            if added:
                break
        #was unable to add to any pre-existing chains, create new region object
        if added == False:
            orderRegions.append(Region(flatchain))

#At this point, we should have a length norders 2D list, each with a list of Region objects.
#Check to see what are the total mu's we've acquired.
print("Classified mu's")
for orderRegions in ordersRegions:
    for region in orderRegions:
        print("\n{} +/- {}".format(region.mu, region.std))

        #Burn in/thin region from end
        region.keep(args.keep, args.thin)

        #compute GR statistic
        region.gelman_rubin()

#concatenate samples into a combined.hdf5 that contains all of the sampled regions.

#make triangle plot for that region

#Specify an amplitude cut. i.e., if the mean amplitude (after some specified period of front-burn in) is below,
# then it is treated as though this region did not exist.
