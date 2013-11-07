#!/usr/bin/env python
"""
Sample using emcee.
"""

import numpy as np
import emcee
import sys
import os
import shutil
#from model import lnprob_old as lnprob
import yaml
import importlib
import model

confname = 'config.yaml' #sys.argv[1]
f = open(confname)
config = yaml.load(f)
f.close()

lnprob = getattr(model, config['lnprob'])

outdir = 'output/' + config['name'] + '/'

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

    ### Do creation of directories here ###
    #Create necessary output directories using os.mkdir, if it does not exist
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        print("Created output directory", outdir)
    else:
        print(outdir, "already exists, overwriting.")

    #Copy config.yaml and SLURM script to this directory
    shutil.copy('config.yaml',outdir + 'config.yaml')
    shutil.copy('run',outdir + 'run')

    # Choose an initial set of positions for the walkers, randomly distributed across a reasonable range of parameters.
    wr = config['walker_ranges']
    temp = np.random.uniform(low=wr['temp'][0], high=wr['temp'][1], size=(nwalkers,))
    logg = np.random.uniform(low=wr['logg'][0], high=wr['logg'][1], size=(nwalkers,))
    Z = np.random.uniform(low=wr['Z'][0], high=wr['Z'][1], size=(nwalkers,))
    #M = np.random.uniform(low=0.1, high = 10, size=(nwalkers,))
    #R = np.random.uniform(low=0.1, high = 10, size=(nwalkers,))
    vsini = np.random.uniform(low=wr['vsini'][0], high=wr['vsini'][1], size=(nwalkers,))
    vz = np.random.uniform(low=wr['vz'][0], high=wr['vz'][1], size=(nwalkers,))
    Av = np.random.uniform(low=wr['Av'][0], high = wr['Av'][1], size=(nwalkers,))
    flux_factor = np.random.uniform(low=wr['flux_factor'][0], high=wr['flux_factor'][1], size=(nwalkers,))
    c0 = np.random.uniform(low=0.9, high = 1.1, size=(nwalkers,))

    p0 = np.array([temp, logg, Z, vsini, vz, Av, flux_factor, c0]).T

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
    np.save(outdir + "chain.npy", sampler.chain)
    #write lnprob to file
    np.save(outdir + "lnprobchain.npy", sampler.lnprobability)

    ### if config['plots'] == True, Call routines to make plots of output ###
    #Histograms of parameters
    #Walker positions as function of step position
    #Samples from the posterior overlaid with the data



if __name__ == "__main__":
    main()
