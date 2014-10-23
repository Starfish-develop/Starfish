import scipy.sparse as sp
import numpy as np
from numpy.linalg import slogdet
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def money(x, y):
    pass

def gauss(x, amp, mu, sigma):
    return amp/np.sqrt(2 * np.pi * sigma**2) * np.exp(-0.5 * (x - mu)**2 / sigma**2)

def line(x, b, m):
    return b + m * x

xs = np.linspace(-20,20)
npoints = len(xs)

#Create a continuum with a gaussian absorption line superimposed. Add Gaussian noise.
ys = line(xs, 10, 0.05) - gauss(xs, amp=15, mu=0, sigma=1) + np.random.normal(size=npoints)

def gauss_func(x0i, x1i, x0v=None, x1v=None, amp=None, mu=None, sigma=None):
    x0 = x0v[x0i]
    x1 = x1v[x1i]
    return amp**2/(2 * np.pi * sigma**2) * np.exp(-((x0 - mu)**2 + (x1 - mu)**2)/(2 * sigma**2))

def Cregion(xs, amp, mu, sigma, var=1):
    '''Create a sparse covariance matrix using identity and block_diagonal'''
    #In the region of the Gaussian, the matrix will be dense, so just create it as `fromfunction`
    #and then later turn it into a sparse matrix with size xs x xs

    #Given mu, and the extent of sigma, estimate the data points that are above, in Gaussian, and below
    n_above = np.sum(xs < (mu - 4 * sigma))
    n_below = np.sum(xs > (mu + 4 * sigma))

    #Create dense matrix and indexes, then convert to lists so that you can pack things in as:

    #csc_matrix((data, ij), [shape=(M, N)])
    #where data and ij satisfy the relationship a[ij[0, k], ij[1, k]] = data[k]

    len_x = len(xs)
    ind_in = (xs >= (mu - 4 * sigma)) & (xs <= (mu + 4 * sigma)) #indices to grab the x values
    len_in = np.sum(ind_in)
    #print(n_above, n_below, len_in)
    #that will be needed to evaluate the Gaussian


    #Create Gaussian matrix fromfunction
    x_gauss = xs[ind_in]
    gauss_mat = np.fromfunction(gauss_func, (len_in,len_in), x0v=x_gauss, x1v=x_gauss,
                                amp=amp, mu=mu, sigma=sigma, dtype=np.int).flatten()

    #Create an index array that matches the Gaussian
    ij = np.indices((len_in, len_in)) + n_above
    ij.shape = (2, -1)

    return sp.csc_matrix((gauss_mat, ij), shape=(len_x,len_x))


def lnprob(p):
    b, m, loga, mu, sigma = p
    if sigma <= 0 or mu < xs[0] or mu > xs[-1]:
        return -np.inf
    else:
        a = 10**loga
        model = line(xs, b, m)
        S = Cregion(xs, amp=a, mu=mu, sigma=sigma) + sp.eye(len(xs))

        sign, logdet = slogdet(S.todense())
        if sign <= 0:
            return -np.inf

        d = ys - model
        lnp = -0.5 * (d.T.dot(spsolve(S, d)) + logdet) - 0.1 * a
        return lnp

def main():
    #print(lnprob(np.array([10, 0.2, 10**5, 0, 1])))
    print(lnprob(np.array([10, 0.2, 1., 0, 10])))
    #print(lnprob(np.array([10, 0.2, 15, 0, 5])))
    #print(lnprob(np.array([10, 0.2, 15, 0, 2])))
    #print(lnprob(np.array([10, 0.2, 15, 0, 5])))

    pass

if __name__=="__main__":
    main()


import emcee


# Initialize the sampler with the chosen specs.
nwalkers = 30
burn_in = 1000
ndim = 5
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

#Declare starting indexes
b = np.random.uniform(low=9, high=11, size=(nwalkers,))
m = np.random.uniform(low=0.0, high=0.1, size=(nwalkers,))
loga = np.random.uniform(low=0.4, high=1.5, size=(nwalkers,))
mu = np.random.uniform(low=-1, high=1, size=(nwalkers,))
sigma = np.random.uniform(low=0.5, high=1.5, size=(nwalkers,))

p0 = np.array([b, m, loga, mu, sigma]).T

pos, prob, state = sampler.run_mcmc(p0, burn_in)

print("Burned in chain")
# Reset the chain to remove the burn-in samples.
sampler.reset()

#Now run for 100 samples
sampler.run_mcmc(pos, 1000, rstate0=state)

import triangle

samples = sampler.flatchain
np.save("samples.npy", samples)
figure = triangle.corner(samples, labels=[r"$b$", r"$m$", r"$\log_{10}(a)$", r"$\mu$", r"$\sigma$"], truths=[10., 0.05, 1.176, 0, 1],
                         quantiles=[0.16, 0.5, 0.84],
                         show_titles=True, title_args={"fontsize": 12})
figure.savefig("plots/triangle.png")
figure.savefig("plots/triangle.eps")
