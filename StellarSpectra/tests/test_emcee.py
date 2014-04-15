#Use emcee as a Metropolis-Hastings so we can avoid a lot of the difficulty of the ensemble sampler for the moment.

import numpy as np
import emcee


#create our lnprob as a multidimensional Gaussian, where icov is C^{-1}
def lnprob(x, mu, icov):
    diff = x-mu
    lnp = -np.dot(diff,np.dot(icov,diff))/2.0
    print("lnp = ", lnp)
    return lnp

ndim = 2

#Create our own parameters for this Gaussian
means = np.array([10, 3])
cov = np.array([[3.0, 0.0],[0.0, 1.0]])
icov = np.linalg.inv(cov)

print("Inverse covariance matrix", icov)

#Jump distribution parameters
MH_cov = np.array([[1.5, 0],[0., 0.7]])

sampler = emcee.MHSampler(MH_cov, ndim, lnprob, args=[means, icov])

pos, prob, state = sampler.run_mcmc(np.array([0, 0]), 5)
print("Samples", sampler.flatchain)
# sampler.reset()

# sampler.run_mcmc(pos, 5)

print("Acceptance fraction", sampler.acceptance_fraction)
#
# import triangle
# import matplotlib.pyplot as plt
#
# samples = sampler.flatchain
# figure = triangle.corner(samples, labels=(r"$\mu_1$", r"$\mu_2$"), quantiles=[0.16, 0.5, 0.84],
#                          show_titles=True, title_args={"fontsize": 12})
# figure.savefig("MH.png")
#
# def plot_walkers(filename, samples, labels=None):
#     ndim = len(samples[0, :])
#     fig, ax = plt.subplots(nrows=ndim, sharex=True)
#     for i in range(ndim):
#         ax[i].plot(samples[:,i])
#         if labels is not None:
#             ax[i].set_ylabel(labels[i])
#     ax[-1].set_xlabel("Sample number")
#     fig.savefig(filename)
#
# plot_walkers("walkers.png", samples, labels=(r"$\mu_1$", r"$\mu_2$"))