#Is it possible to update lnprob args without clearing all of the samples?

import numpy as np
import emcee
from emcee.utils import MPIPool
import sys


class function_wrapper(object):
    """
    This is a hack of a hack to make the likelihood function pickleable and update-able when ``args``
    or ``kwargs`` are also included.

    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            print("emcee: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise

def update_args(sampler, *args, **kwargs):
    '''
    Take a sampler object and update it's args.
    '''
    sampler.args = args
    sampler.kwargs = kwargs
    sampler.lnprobfn = function_wrapper(lnprob, args=args, kwargs=kwargs)



xs = np.linspace(0, 10, num=20)
ys = 5 + 10 * xs + np.random.normal(size=20)

def lnprob(p, b):
    m = p[0]
    chi2 = np.sum((ys - (b + m*xs))**2)
    #print("{:.1f} ".format(b),end="")
    return -0.5 * chi2

pool = MPIPool()

if not pool.is_master():
#   Wait for instructions from the master process.
    pool.wait() #all execution takes place here
    sys.exit(0) #this happens at the very end of the run.

nwalkers = 10
burn_in = 10
ndim = 1
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=np.array([-10]))#, pool=pool)

#Declare starting indexes
m = np.random.uniform(low=8, high=12, size=(nwalkers,))
p0 = np.array([m]).T
pos, prob, state = sampler.run_mcmc(p0, burn_in)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000, rstate0=state)
print(np.mean(sampler.flatchain))



#Try updating args
update_args(sampler, 5)
pos, prob, state = sampler.run_mcmc(pos, 1000, rstate0=state)
#Burn in again
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000, rstate0=state)
print(np.mean(sampler.flatchain))


pool.close()

##How to use ProgressBar
#from progressbar import ProgressBar, Percentage, Bar, ETA
#pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=1000).start()
#index = 0
#for result in sampler.sample(pos, prob, state, iterations=1000):
#    index += 1
#    pbar.update(index)
#pbar.finish()