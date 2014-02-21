import scipy.sparse as sp
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def gauss(x, amp, mu, sigma):
    return amp/np.sqrt(2 * np.pi * sigma**2) * np.exp(-0.5 * (x - mu)**2 / sigma**2)

def line(x, b, m):
    return b + m * x

xs = np.linspace(-10,10)
npoints = len(xs)

#Create a continuum with a gaussian absorption line superimposed. Add Gaussian noise.
ys = line(xs, 10, 0.2) - gauss(xs, amp=15, mu=0, sigma=1) + np.random.normal(size=npoints)

def plot_line():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs,ys)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    fig.savefig("plots/correlated_gaussian.png")
plot_line()


def Cinv(amp, mu, sigma, var=1):
    '''Return the inverse of the covariance matrix as a sparse array'''
    S = sp.dok_matrix((npoints, npoints), dtype=np.float64)
    for i in range(npoints):
        for j in range(npoints):
            x0 = xs[i]
            x1 = xs[j]
            if np.abs(x0) < 4*sigma and np.abs(x1) < 4*sigma:
                if i == j:
                    S[i,j] = amp**2/(2 * np.pi * sigma**2) * np.exp(-((x0 - mu)**2 + (x1 - mu)**2)/(2 * sigma**2)) + var
                else:
                    S[i,j] = amp**2/(2 * np.pi * sigma**2) * np.exp(-((x0 - mu)**2 + (x1 - mu)**2)/(2 * sigma**2))
            elif i == j:
                S[i,j] = var
    return inv(S.todense())

def chi2(b, m, aG, muG, sigmaG):
    model = line(xs, b, m)
    diff = ys - model
    diff.shape = (-1, 1)
    Sinv = Cinv(amp=aG, mu=muG, sigma=sigmaG)
    result = Sinv.dot(diff)
    chi_val = diff.T.dot(result)
    return chi_val[0,0]

def lnprob(p):
    return - chi2(*p) - p[3]

def main():
    pass
    #chi2(10, 0.2, 15, 0, 1)

if __name__=="__main__":
    main()


import emcee


# Initialize the sampler with the chosen specs.
nwalkers = 30
burn_in = 1000
ndim = 5
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4)

#Declare starting indexes
m = np.random.uniform(low=5, high=15, size=(nwalkers,))
b = np.random.uniform(low=-0.1, high=0.3, size=(nwalkers,))
a = np.random.uniform(low=0, high=50, size=(nwalkers,))
mu = np.random.uniform(low=-5, high=5, size=(nwalkers,))
sigma = np.random.uniform(low=0.1, high=3, size=(nwalkers,))

p0 = np.array([m, b, a, mu, sigma]).T

pos, prob, state = sampler.run_mcmc(p0, burn_in)

print("Burned in chain")
# Reset the chain to remove the burn-in samples.
sampler.reset()

#Now run for 100 samples
sampler.run_mcmc(pos, 1000, rstate0=state)

import triangle

samples = sampler.flatchain
figure = triangle.corner(samples, labels=[r"$b$", r"$m$", r"$a$", r"$\mu$", r"$\sigma$"], truths=[10., 0.2, 15, 0, 1],
                         quantiles=[0.16, 0.5, 0.84],
                         show_titles=True, title_args={"fontsize": 12})
figure.savefig("plots/triangle.png")