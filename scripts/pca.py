#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser(description="Create and manipulate a PCA decomposition of the synthetic spectral library.")
parser.add_argument("--create", action="store_true", help="Create a PCA decomposition.")

parser.add_argument("--plot", choices=["reconstruct", "priors", "emcee",
                                       "emulator"], help="reconstruct: plot the original synthetic spectra vs. the PCA reconstructed spectra.\n priors: plot the chosen priors on the parameters emcee: plot the triangle diagram for the result of the emcee optimization. emulator: plot weight interpolations")

parser.add_argument("--optimize", choices=["fmin", "emcee"], help="Optimize the emulator using either a downhill simplex algorithm or the emcee ensemble sampler algorithm.")

parser.add_argument("--resume", action="store_true", help="Designed to be used with the --optimize flag to continue from the previous set of parameters. If this is left off, the chain will start from your initial guess specified in config.yaml.")

parser.add_argument("--samples", type=int, default=100, help="Number of samples to run the emcee ensemble sampler.")
args = parser.parse_args()

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import Starfish
from Starfish import emulator
from Starfish.grid_tools import HDF5Interface
from Starfish.emulator import PCAGrid, Gprior, Glnprior
from Starfish.covariance import Sigma


if args.create:
    myHDF5 = HDF5Interface()
    my_pca = PCAGrid.create(myHDF5)
    my_pca.write()

if args.plot == "reconstruct":
    my_HDF5 = HDF5Interface()
    my_pca = PCAGrid.open()

    recon_fluxes = my_pca.reconstruct_all()

    # we need to apply the same normalization to the synthetic fluxes that we
    # used for the reconstruction
    fluxes = np.empty((my_pca.M, my_pca.npix))
    for i, spec in enumerate(my_HDF5.fluxes):
        fluxes[i,:] = spec

    # Normalize all of the fluxes to an average value of 1
    # In order to remove uninteresting correlations
    fluxes = fluxes/np.average(fluxes, axis=1)[np.newaxis].T

    data = zip(my_HDF5.grid_points, fluxes, recon_fluxes)

    # Define the plotting function
    def plot(data):
        par, real, recon = data
        fig, ax = plt.subplots(nrows=2, figsize=(8, 8))
        ax[0].plot(my_pca.wl, real)
        ax[0].plot(my_pca.wl, recon)
        ax[0].set_ylabel(r"$f_\lambda$")

        ax[1].plot(my_pca.wl, real - recon)
        ax[1].set_xlabel(r"$\lambda$ [AA]")
        ax[1].set_ylabel(r"$f_\lambda$")

        fmt = "=".join(["{:.2f}" for i in range(len(Starfish.parname))])
        name = fmt.format(*[p for p in par])
        ax[0].set_title(name)
        fig.savefig(Starfish.config["plotdir"] + "PCA_" + name + ".png")
        plt.close("all")

    p = mp.Pool(mp.cpu_count())
    p.map(plot, data)


if args.plot == "priors":
    # Read the priors on each of the parameters from Starfish config.yaml
    priors = Starfish.PCA["priors"]
    for i,par in enumerate(Starfish.parname):
        s, r = priors[i]
        mu = s/r
        x = np.linspace(0.01, 2 * mu)
        prob = Gprior(x, s, r)
        plt.plot(x, prob)
        plt.xlabel(par)
        plt.ylabel("Probability")
        plt.savefig(Starfish.config["plotdir"] + "prior_" + par + ".png")
        plt.close("all")

# If we're doing optimization, period, set up some variables and the lnprob
if args.optimize:
    my_pca = emulator.PCAGrid.open()
    PhiPhi = np.linalg.inv(emulator.skinny_kron(my_pca.eigenspectra, my_pca.M))
    priors = Starfish.PCA["priors"]

    def lnprob(p, fmin=False):
        '''
        :param p: Gaussian Processes hyper-parameters
        :type p: 1D np.array

        Calculate the lnprob using Habib's posterior formula for the emulator.
        '''

        # We don't allow negative parameters.
        if np.any(p < 0.):
            if fmin:
                return 1e99
            else:
                return -np.inf

        lambda_xi = p[0]
        hparams = p[1:].reshape((my_pca.m, -1))

        # Calculate the prior for parname variables
        # We have two separate sums here, since hparams is a 2D array
        # hparams[:, 0] are the amplitudes, so we index i+1 here
        lnpriors = 0.0
        for i in range(0, len(Starfish.parname)):
            lnpriors += np.sum(Glnprior(hparams[:, i+1], *priors[i]))

        h2params = hparams**2
        #Fold hparams into the new shape
        Sig_w = Sigma(my_pca.gparams, h2params)

        C = (1./lambda_xi) * PhiPhi + Sig_w

        sign, pref = np.linalg.slogdet(C)

        central = my_pca.w_hat.T.dot(np.linalg.solve(C, my_pca.w_hat))

        lnp = -0.5 * (pref + central + my_pca.M * my_pca.m * np.log(2. * np.pi)) + lnpriors

        # Negate this when using the fmin algorithm
        if fmin:
            print("lambda_xi", lambda_xi)
            for row in hparams:
                print(row)
            print()
            print(lnp)

            return -lnp
        else:
            return lnp


if args.optimize == "fmin":

    if args.resume:
        p0 = np.load("eparams.npy")

    else:
        amp = 100.
        # Use the mean of the gamma distribution to start
        eigpars = np.array([amp] + [s/r for s,r in priors])
        p0 = np.hstack((np.array([1., ]), #lambda_xi
        np.hstack([eigpars for i in range(my_pca.m)])))

    from scipy.optimize import fmin
    func = lambda p : lnprob(p, fmin=True)
    result = fmin(func, p0, maxiter=10000, maxfun=10000)
    print(result)
    np.save("eparams.npy", result)

if args.optimize == "emcee":

    import emcee

    ndim = 1 + (1 + len(Starfish.parname)) * my_pca.m
    nwalkers = 4 * ndim # about the minimum per dimension we can get by with

    # Assemble p0 based off either a guess or the previous state of walkers
    if args.resume:
        p0 = np.load("walkers_start.npy")
    else:
        p0 = []
        # p0 is a (nwalkers, ndim) array
        amp = [10.0, 150]

        p0.append(np.random.uniform(0.01, 1.0, nwalkers))
        for i in range(my_pca.m):
            p0 +=   [np.random.uniform(amp[0], amp[1], nwalkers)]
            for s,r in priors:
                # Draw randomly from the gamma priors
                p0 += [np.random.gamma(s, r, nwalkers)]

        p0 = np.array(p0).T


    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=mp.cpu_count())

    # burn in
    pos, prob, state = sampler.run_mcmc(p0, args.samples)
    sampler.reset()
    print("Burned in")

    # actual run
    pos, prob, state = sampler.run_mcmc(pos, args.samples)

    # Save the last position of the walkers
    np.save("walkers_start.npy", pos)
    np.save("eparams_walkers.npy", sampler.flatchain)


if args.plot == "emcee":
    #Make a triangle plot of the samples
    my_pca = emulator.PCAGrid.open()
    flatchain = np.load("eparams_walkers.npy")
    import triangle

    # figure out how many separate triangle plots we need to make
    npar = len(Starfish.parname) + 1
    labels = ["amp"] + Starfish.parname

    # Make a histogram of lambda xi
    plt.hist(flatchain[:,0], histtype="step", normed=True)
    plt.title(r"$\lambda_\xi$")
    plt.xlabel(r"$\lambda_\xi$")
    plt.ylabel("prob")
    plt.savefig(Starfish.config["plotdir"] + "triangle_lambda_xi.png")

    # Make a triangle plot for each eigenspectrum independently
    for i in range(my_pca.m):
        start = 1 + i * npar
        end = 1 + (i + 1) * npar
        figure = triangle.corner(flatchain[:, start:end], quantiles=[0.16, 0.5, 0.84],
            plot_contours=True, plot_datapoints=False, show_titles=True, labels=labels)
        figure.savefig(Starfish.config["plotdir"] + "triangle_{}.png".format(i))
