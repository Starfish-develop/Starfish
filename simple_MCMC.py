import numpy as np
import emcee
from emcee.utils import MPIPool
import sys
from simple_model import myLnprob


pool = MPIPool()

if not pool.is_master():
#   Wait for instructions from the master process.
    pool.wait()
    sys.exit(0) #this is at the very end of the run.

# Initialize the sampler with the chosen specs.
nwalkers = 10
burn_in = 100
sampler = emcee.EnsembleSampler(nwalkers, 2, myLnprob.lnprob, pool=pool)


m = np.random.uniform(low=0, high=3, size=(nwalkers,))
b = np.random.uniform(low=0, high=3, size=(nwalkers,))

p0 = np.array([m, b]).T

pos, prob, state = sampler.run_mcmc(p0, burn_in)

print("Burned in chain")
# Reset the chain to remove the burn-in samples.
sampler.reset()
