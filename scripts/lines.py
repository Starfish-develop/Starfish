#!/usr/bin/env python

import numpy as np
import Starfish
from Starfish.single import args, lnprob
import multiprocessing as mp


if args.sample:
    print(lnprob(np.array([6350, 4.1, -0.3, -4.8, 5.3, -12.8])))
    # # Use vanilla emcee to do the sampling
    from emcee import EnsembleSampler
    #
    ndim = 6
    nwalkers = 4 * ndim
    #
    # # Load values from config file.
    # # Add scatter in
    #
    p0 = np.array([ np.random.uniform(6300, 6600, nwalkers),
                    np.random.uniform(3.9, 4.5, nwalkers),
                    np.random.uniform(-0.5, -0.01, nwalkers),
                    np.random.uniform(-5.5, -4.5, nwalkers),
                    np.random.uniform(4.0, 5.0, nwalkers),
                    np.random.uniform(-12.82, -12.78, nwalkers)]).T

    sampler = EnsembleSampler(nwalkers, ndim, lnprob, threads=mp.cpu_count())
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
