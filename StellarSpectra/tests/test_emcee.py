#Use emcee as a Metropolis-Hastings so we can avoid a lot of the difficulty of the ensemble sampler for the moment.

import numpy as np
import emcee



#create our lnprob as a multidimensional Gaussian, where icov is C^{-1}
def lnprob(x, mu, icov):
    diff = x-mu
    return -np.dot(diff,np.dot(icov,diff))/2.0

ndim = 2

#Choose some random mean for these points
means = np.random.rand(ndim)

#This creates a symmetric covariance matrix
cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim)) #create C
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov,cov)

#This takes the inverse
icov = np.linalg.inv(cov)

MH_cov = np.array([[0.1, 0],[0., 0.1]])

sampler = emcee.MHSampler(MH_cov, ndim, lnprob, args=[means, icov])

sampler.run_mcmc(np.array([0, 0]), 100)

print(sampler.flatchain)