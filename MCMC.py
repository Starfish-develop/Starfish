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
import yaml

confname = 'config.yaml' #sys.argv[1]
f = open(confname)
config = yaml.load(f)
f.close()


def generate_nuisance_params():
    '''convenience method for generating walker starting positions for nuisance parameters'''
    norders = len(config['orders'])
    #determine as (norder, ncoeff) array, aka (norder, -1) then reshape as necessary
    pass


def main():
    ndim = config['ndim']
    nwalkers = config['nwalkers']

    if config['MPI']:
        from emcee.utils import MPIPool
        # Initialize the MPI-based pool used for parallelization.
        pool = MPIPool(debug=True)
        print("Running with MPI")

        if not pool.is_master():
        #   Wait for instructions from the master process.
            pool.wait()
            sys.exit(0) #this is at the very end of the run.

        # Initialize the sampler with the chosen specs.
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=config['threads'])

    # Choose an initial set of positions for the walkers, randomly distributed across a reasonable range of parameters.
    temp = np.random.uniform(low=5500, high=6300, size=(nwalkers,))
    logg = np.random.uniform(low=3.0, high=3.7, size=(nwalkers,))
    #M = np.random.uniform(low=0.1, high = 10, size=(nwalkers,))
    #R = np.random.uniform(low=0.1, high = 10, size=(nwalkers,))
    vsini = np.random.uniform(low=35, high=55, size=(nwalkers,))
    vz = np.random.uniform(low=27, high=29.5, size=(nwalkers,))
    #Av = np.random.uniform(low=1, high = 5, size=(nwalkers,))
    flux_factor = np.random.uniform(low=1.e-28, high=1.e-27, size=(nwalkers,))
    #c0_21 = np.random.uniform(low=0.9, high = 1.1, size=(nwalkers,))
    c1_21 = np.random.uniform(low=-0.1, high=0.1, size=(nwalkers,))
    c2_21 = np.random.uniform(low=-0.1, high=0.1, size=(nwalkers,))
    #c0_22 = np.random.uniform(low=0.9, high = 1.1, size=(nwalkers,))
    #c1_22 = np.random.uniform(low=-0.1, high = 0.1, size=(nwalkers,))
    #c2_22 = np.random.uniform(low=-0.1, high = 0.1, size=(nwalkers,))
    #c0_23 = np.random.uniform(low=0.9, high = 1.1, size=(nwalkers,))
    #c1_23 = np.random.uniform(low=-0.1, high = 0.1, size=(nwalkers,))
    #c2_23 = np.random.uniform(low=-0.1, high = 0.1, size=(nwalkers,))

    p0 = np.array([temp, logg, vsini, vz, flux_factor, c1_21, c2_21]).T#,c0_22,c1_22,c2_22,c0_23,c1_23,c2_23]).T

    # Burn-in.
    pos, prob, state = sampler.run_mcmc(p0, config['burn_in'])

    print("Burned in chain")
    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    # Starting from the final position in the burn-in chain, sample for 1000
    # steps.
    #f = open("chain.dat", "w")
    #f.close()
    sampler.run_mcmc(pos, config['iterations'], rstate0=state)
    #    position = result[0]
    #    f = open("chain.dat", "a")
    #    for k in range(position.shape[0]):
    #        f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k])))
    #    f.close()

    if config['MPI']:
        pool.close()

    # Print out the mean acceptance fraction. In general, acceptance_fraction
    # has an entry for each walker so, in this case, it is a 250-dimensional
    # vector.
    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

    #write chain to file
    np.save("output/chain.npy", sampler.chain)
    #write lnprob to file
    np.save("output/lnprobchain.npy", sampler.lnprobability)


if __name__ == "__main__":
    main()
