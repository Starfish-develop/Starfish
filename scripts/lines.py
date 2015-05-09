#!/usr/bin/env python

import numpy as np
import Starfish
from Starfish.single import args, lnprob
import multiprocessing as mp


if args.sample:
    print(lnprob(np.array([6302, 4.38, 0.1, -39.54, 5.6, -12.221])))
    # # Use vanilla emcee to do the sampling
    from emcee import EnsembleSampler
    #
    ndim = 6
    nwalkers = 4 * ndim
    #
    # # Load values from config file.
    # # Add scatter in
    #
    p0 = np.array([ np.random.uniform(6200, 6400, nwalkers),
                    np.random.uniform(4.0, 4.49, nwalkers),
                    np.random.uniform(-0.2, -0.1, nwalkers),
                    np.random.uniform(-5., -4., nwalkers),
                    np.random.uniform(4.0, 6.0, nwalkers),
                    np.random.uniform(-12.81, -12.80, nwalkers)]).T

    sampler = EnsembleSampler(nwalkers, ndim, lnprob, threads=mp.cpu_count()-1)
    #
    # # burn in
    pos, prob, state = sampler.run_mcmc(p0, args.samples)
    sampler.reset()
    print("Burned in")
    #
    # actual run
    pos, prob, state = sampler.run_mcmc(pos, args.samples)

    # Save the last position of the walkers
    np.save("walkers_emcee.npy", pos)
    np.save("eparams_emcee.npy", sampler.flatchain)
