import numpy as np
import emcee
from emcee.utils import MPIPool
import sys
from StellarSpectra.simple_model import Data, LineModel

#Load data
wl = np.load("data/Fake/line.wls.npy")
fl = np.load("data/Fake/line.fls.npy")

#Initialize model and data objects in global namespace
myData = Data(wl, fl)
myModel = LineModel(wl, np.array([1.5, 1.5]))

#Create lnprob in global scope
def lnprob(param):
    myModel.set_params(param)
    return np.sum((myData.y - myModel.y)**2)

pool = MPIPool()

if not pool.is_master():
#   Wait for instructions from the master process.
    pool.wait()
    sys.exit(0) #this is at the very end of the run.

# Initialize the sampler with the chosen specs.
nwalkers = 20
burn_in = 100
ndim = 2
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

#Declare starting indexes
m = np.random.uniform(low=0, high=3, size=(nwalkers,))
b = np.random.uniform(low=0, high=3, size=(nwalkers,))

p0 = np.array([m, b]).T

pos, prob, state = sampler.run_mcmc(p0, burn_in)

print("Burned in chain")
# Reset the chain to remove the burn-in samples.
sampler.reset()

#Now run for 100 samples
sampler.run_mcmc(pos, 100, rstate0=state)

pool.close()

import triangle

samples = sampler.flatchain
figure = triangle.corner(samples, labels=["$m$", "$b$"], truths=[0.0, 1.0], quantiles=[0.16, 0.5, 0.84],
                         show_titles=True, title_args={"fontsize": 12})
figure.savefig("plots/triangle.png")