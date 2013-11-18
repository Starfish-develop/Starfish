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
import plot_MCMC

if len(sys.argv) > 1:
    confname= sys.argv[1]
else:
    confname = 'config.yaml'
f = open(confname)
config = yaml.load(f)
f.close()


nwalkers = config['nwalkers']
ncoeff = config['ncoeff']
norders = len(config['orders'])
wr = config['walker_ranges']

if (config['lnprob'] == 'lnprob_gaussian_marg') or (config['lnprob'] == 'lnprob_lognormal_marg'):
    ndim = config['nparams'] + norders
if (config['lnprob'] == "lnprob_lognormal") or (config['lnprob'] == "lnprob_gaussian"):
    ndim = config['nparams'] + ncoeff * norders

lnprob = getattr(model, config['lnprob'])

outdir = 'output/' + config['name'] + '/'

def generate_nuisance_params():
    '''convenience method for generating walker starting positions for nuisance parameters.
    Reads number of orders from config, type of lnprob and generates c0, c1, c2, c3 locations,
    or just c0 locations if lnprob_marg'''
    norders = len(config['orders'])
    c0s = np.random.uniform(low=wr['c0'][0], high = wr['c0'][1], size=(norders, nwalkers))
    if (config['lnprob'] == 'lnprob_gaussian_marg') or (config['lnprob'] == 'lnprob_lognormal_marg'):
        return c0s

    if (config['lnprob'] == "lnprob_lognormal") or (config['lnprob'] == "lnprob_gaussian"):
        #do this for each order. create giant array for cns, then do a stride on every c0 to replace them.
        cs = np.random.uniform(low=wr['cs'][0], high = wr['cs'][1], size=(ncoeff*norders, nwalkers))
        cs[::ncoeff] = c0s
        return cs

def main():

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
        os.makedirs(outdir, exist_ok=True)
        print("Created output directory", outdir)
    else:
        print(outdir, "already exists, overwriting.")

    #Copy config.yaml and SLURM script to this directory
    shutil.copy('config.yaml', outdir + 'config.yaml')
    shutil.copy('run', outdir + 'run')

    # Choose an initial set of positions for the walkers, randomly distributed across a reasonable range of parameters.

    temp = np.random.uniform(low=wr['temp'][0], high=wr['temp'][1], size=(nwalkers,))
    logg = np.random.uniform(low=wr['logg'][0], high=wr['logg'][1], size=(nwalkers,))
    Z = np.random.uniform(low=wr['Z'][0], high=wr['Z'][1], size=(nwalkers,))
    #M = np.random.uniform(low=0.1, high = 10, size=(nwalkers,))
    #R = np.random.uniform(low=0.1, high = 10, size=(nwalkers,))
    vsini = np.random.uniform(low=wr['vsini'][0], high=wr['vsini'][1], size=(nwalkers,))
    vz = np.random.uniform(low=wr['vz'][0], high=wr['vz'][1], size=(nwalkers,))
    Av = np.random.uniform(low=wr['Av'][0], high = wr['Av'][1], size=(nwalkers,))
    flux_factor = np.random.uniform(low=wr['flux_factor'][0], high=wr['flux_factor'][1], size=(nwalkers,))
    cs = generate_nuisance_params()

    p0 = np.vstack((np.array([temp, logg, Z, vsini, vz, Av, flux_factor]), cs)).T #Stack cs onto the end

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
    if config['plots'] == True:
        plot_MCMC.auto_hist_param(sampler.flatchain)
        plot_MCMC.hist_nuisance_param(sampler.flatchain)
    #Histograms of parameters
    #Walker positions as function of step position
    #Samples from the posterior overlaid with the data


if __name__ == "__main__":
    main()
