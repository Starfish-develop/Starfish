#!/usr/bin/env python
"""
Sample the discretized grid using emcee.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import emcee
import sys
from model import lnprob
#from emcee.utils import MPIPool


def main():
    #11 dimensional model, 200 walkers
    ndim = 6
    nwalkers = 150

    # Choose an initial set of positions for the walkers, randomly distributed across a reasonable range of parameters.
    temp = np.random.uniform(low=4200, high = 6800, size=(nwalkers,))
    logg = np.random.uniform(low=0.0, high=6.0, size=(nwalkers,))
    #M = np.random.uniform(low=0.1, high = 10, size=(nwalkers,))
    #R = np.random.uniform(low=0.1, high = 10, size=(nwalkers,))
    #Av = np.random.uniform(low=0, high = 8, size=(nwalkers,))
    vsini = np.random.uniform(low=30, high = 70, size=(nwalkers,))
    vz = np.random.uniform(low=25, high = 32, size=(nwalkers,))
    c0 = np.random.uniform(low=1e27, high = 4e27, size=(nwalkers,))
    c1 = np.random.uniform(low=-0.01, high = 0.01, size=(nwalkers,))
    #c2 = np.random.uniform(low=-0.01, high = 0.01, size=(nwalkers,))
    #c3 = np.random.uniform(low=-0.01, high = 0.01, size=(nwalkers,))
    #c4 = np.random.uniform(low=-0.01, high = 0.01, size=(nwalkers,))

    p0 = np.array([temp,logg,vsini,vz,c0,c1]).T

    # Initialize the sampler with the chosen specs.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,threads=4)

    # Burn-in.
    pos, prob, state = sampler.run_mcmc(p0, 50)

    print("Burned in chain")
    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    # Starting from the final position in the burn-in chain, sample for 1000
    # steps.
    #f = open("chain.dat", "w")
    #f.close()
    sampler.run_mcmc(pos, 100, rstate0=state)
    #    position = result[0]
    #    f = open("chain.dat", "a")
    #    for k in range(position.shape[0]):
    #        f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k])))
    #    f.close()

    # Print out the mean acceptance fraction. In general, acceptance_fraction
    # has an entry for each walker so, in this case, it is a 250-dimensional
    # vector.
    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

    #write flatchain to file
    np.save("flatchain.npy", sampler.flatchain)
    #write lnprob to file
    np.save("lnprobchain.npy", sampler.flatlnprobability)

if __name__=="__main__":
    main()
