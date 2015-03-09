# Parallel implementation for sampling a multi-order echelle spectrum.
# Because the likelihood calculation is independent for each order, the
# runtime is essentially constant regardless of how large a spectral range is used.

# Additionally, one could use this to fit multiple stars at once.

import argparse
parser = argparse.ArgumentParser(prog="parallel.py", description="Run Starfish fitting model in parallel.")
parser.add_argument("-r", "--run_index", help="Which run (of those running concurrently) is this? All data will be written into this directory, overwriting any that exists.")
parser.add_argument("-p", "--perturb", type=float, help="Randomly perturb the starting position of the chain, as a multiple of the jump parameters.")
parser.add_argument("--optimize", choices=["Phi", "Theta"], help="Optimize the parameters using fmin.")
parser.add_argument("--sample", choices=["Phi", "Theta"], help="Sample the parameters.")
args = parser.parse_args()

from multiprocessing import Process, Pipe
import os
import numpy as np

import Starfish
import Starfish.grid_tools
from Starfish.samplers import StellarSampler, NuisanceSampler
from Starfish.spectrum import DataSpectrum, Mask, ChebyshevSpectrum
from Starfish.emulator import Emulator
import Starfish.constants as C
from Starfish.covariance import get_dense_C, make_k_func

from scipy.special import j1
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet
from astropy.stats.funcs import sigma_clip

import gc
import logging

from itertools import chain
from collections import deque
from operator import itemgetter
import yaml
import shutil

# If we are sampling, then we need to setup output directories to store the samples and other output products. If we're sampling, we probably want to be running multiple chains at once, and so we have to set things up so that they don't conflict.
if args.sample:
    base = Starfish.outdir + Starfish.name + "run{:0>2}/"
    # This code is necessary for multiple simultaneous runs on a cluster
    # so that different runs do not write into the same output directory
    if args.run_index == None:
        run_index = 0
        while os.path.exists(base.format(run_index)):
            print(base.format(run_index), "exists")
            run_index += 1
        outdir = base.format(run_index)

    else:
        run_index = args.run_index
        outdir = base.format(run_index)
        #Delete this outdir, if it exists
        if os.path.exists(outdir):
            print("Deleting", outdir)
            shutil.rmtree(outdir)

    print("Creating ", outdir)
    os.makedirs(outdir)

    # Copy yaml file to outdir for archiving purposes
    shutil.copy("config.yaml", outdir + "/config.yaml")

    for model_number in range(len(DataSpectra)):
        for order in config['orders']:
            order_dir = "{}{}/{}".format(outdir, model_number, order)
            print("Creating ", order_dir)
            os.makedirs(order_dir)

# Otherwise, we'll be optimizing and we can write everything into the current working directory
else:
    outdir = ""

# Determine how many filenames we have.
# Always load as a list, even len == 1.
# If there are multiple datasets, this list will be longer than length 1
orders = Starfish.data["orders"] # list of which orders to fit
order_keys = np.arange(len(orders)) # list of keys from 0 to (norders - 1)
DataSpectra = [DataSpectrum.open(file, orders=orders) for file in Starfish.data["files"]]

# Number of different data sets we are fitting. Used for indexing purposes.
spectra_keys = np.arange(len(DataSpectra))

#Instruments are provided as one per dataset
Instruments = [eval("Starfish.grid_tools." + inst)() for inst in Starfish.data["instruments"]]

masks = config.get("mask", None)
if masks is not None:
    for mask, dataSpec in zip(masks, DataSpectra):
        myMask = Mask(mask, orders=orders)
        dataSpec.add_mask(myMask.masks)

# Set up the logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", filename="{}log.log".format(
    outdir), level=logging.DEBUG, filemode="w", datefmt='%m/%d/%Y %I:%M:%S %p')


def perturb(startingDict, jumpDict, factor=3.):
    '''
    Given a starting parameter dictionary loaded from a config file, perturb the
    values as a multiple of the jump distribution. This is designed so that
    not all chains start at exactly the same place.

    Modifies the startingDict
    '''
    for key in startingDict.keys():
        startingDict[key] += factor * np.random.normal(loc=0, scale=jumpDict[key])


stellar_Starting = config['stellar_params']
stellar_tuple = C.dictkeys_to_tuple(stellar_Starting)
# go through each item in stellar_tuple, and assign the appropriate covariance to it
stellar_MH_cov = np.array([float(config["stellar_jump"][key]) for key in stellar_tuple])**2 \
                 * np.identity(len(stellar_Starting))

fix_logg = config.get("fix_logg", None)

# Updating specific covariances to speed mixing
if config["use_cov"]:
    # Use an emprically determined covariance matrix to for the jumps.
    pass

def info(title):
    '''
    Print process information useful for debugging.
    '''
    print(title)
    print('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
    print('process id:', os.getpid())


class OrderModel:
    def __init__(self, debug=False):
        '''
        This object contains all of the variables necessary for the partial
        lnprob calculation for one echelle order. It is designed to first be
        instantiated within the main processes and then forked to other
        subprocesses. Once operating in the subprocess, the variables specific
        to the order are loaded with an `INIT` message call, which tells which key
        to initialize on in the `self.initialize()`.
        '''
        self.lnprob = -np.inf
        self.lnprob_last = -np.inf

        self.func_dict = {"INIT": self.initialize,
                          "DECIDE": self.decide_stellar,
                          "INST": self.instantiate,
                          "LNPROB": self.stellar_lnprob,
                          "GET_LNPROB": self.get_lnprob,
                          "FINISH": self.finish
                          }

        self.debug = debug

    def initialize(self, key):
        '''
        Initialize the OrderModel to the correct chunk of data (echelle order).

        :param key: (spectrum_id, order_key)
        :param type: (int, int)

        This should only be called after all subprocess have been forked.
        '''

        self.id = key
        self.spectrum_id, self.order_key = self.id

        self.logger.info("Initializing model on Spectrum {}, order {}.".format(self.spectrum_id, self.order_key))

        self.instrument = Instruments[self.spectrum_id]
        self.DataSpectrum = DataSpectra[self.spectrum_id]
        self.wl = self.DataSpectrum.wls[self.order_key]
        self.fl = self.DataSpectrum.fls[self.order_key]
        self.sigma = self.DataSpectrum.sigmas[self.order_key]
        self.npoints = len(self.wl)
        self.mask = self.DataSpectrum.masks[self.order_key]
        self.order = self.DataSpectrum.orders[self.order_key]

        self.logger = logging.getLogger("{} {}".format(self.__class__.__name__, self.order))
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.npoly = config["cheb_degree"]
        self.ChebyshevSpectrum = ChebyshevSpectrum(self.DataSpectrum, self.order_key, npoly=self.npoly)
        self.resid_deque = deque(maxlen=500) #Deque that stores the last residual spectra, for averaging
        self.counter = 0

        self.Emulator = Emulator.open(config["PCA_path"]) # Returns mu and var vectors
        self.Emulator.determine_chunk_log(self.wl) # Truncates the grid to this wl format, power of 2

        pg = self.Emulator.pca

        self.wl_FFT = pg.wl
        self.ncomp = pg.ncomp

        self.EIGENSPECTRA = np.vstack((pg.flux_mean[np.newaxis,:], pg.flux_std[np.newaxis,:], pg.eigenspectra))

        self.min_v = self.Emulator.min_v
        self.ss = np.fft.rfftfreq(len(self.wl_FFT), d=self.min_v)
        self.ss[0] = 0.01 # junk so we don't get a divide by zero error

        self.eigenspectra = np.empty((self.ncomp, self.npoints))
        self.flux_mean = np.empty((self.npoints,))
        self.flux_std = np.empty((self.npoints,))
        self.mus, self.vars = None, None
        self.C_GP = None
        self.data_mat = None

        self.sigma_matrix = self.sigma**2 * np.eye(self.npoints)

        self.prior = 0.0 # Modified and set by NuisanceSampler.lnprob
        self.nregions = 0
        self.exceptions = []

        #TODO: perturb
        #if args.perturb:
            #perturb(stellar_Starting, config["stellar_jump"], factor=args.perturb)

        cheb_MH_cov = float(config["cheb_jump"])**2 * np.ones((self.npoly,))
        cheb_tuple = ("logc0",)
        # add in new coefficients
        for i in range(1, self.npoly):
            cheb_tuple += ("c{}".format(i),)
        # set starting position to 0
        cheb_Starting = {k:0.0 for k in cheb_tuple}

        # Design cov starting
        cov_Starting = config['cov_params']
        cov_tuple = C.dictkeys_to_cov_global_tuple(cov_Starting)
        cov_MH_cov = np.array([float(config["cov_jump"][key]) for key in cov_tuple])**2

        nuisance_MH_cov = np.diag(np.concatenate((cheb_MH_cov, cov_MH_cov)))
        nuisance_starting = {"cheb": cheb_Starting, "cov": cov_Starting, "regions":{}}

        # Because this initialization is happening on the subprocess, I think
        # the random state should be fine.

        # Update the outdir based upon id
        self.noutdir = outdir + "{}/{}/".format(self.spectrum_id, self.order)

        # Create the nuisance parameter sampler to run independently
        self.sampler = NuisanceSampler(OrderModel=self, starting_param_dict=nuisance_starting, cov=nuisance_MH_cov, debug=True, outdir=self.noutdir, order=self.order)
        self.p0 = self.sampler.p0

        # Udpate the nuisance parameters to the starting values so that we at
        # least have a self.data_mat
        self.logger.info("Updating nuisance parameter data products to starting values.")
        self.update_nuisance(nuisance_starting)
        self.lnprob = None

    def instantiate(self, *args):
        '''
        Clear the old NuisanceSampler, instantiate the regions using the stored
        residual spectra, and create a new NuisanceSampler.
        '''

        # threshold for sigma clipping
        sigma=config["sigma_clip"]

        # array that specifies if a pixel is already covered.
        # to start, it should be all False
        covered = np.zeros((self.npoints,), dtype='bool')

        #average all of the spectra in the deque together
        residual_array = np.array(self.resid_deque)
        if len(self.resid_deque) == 0:
            raise RuntimeError("No residual spectra stored yet.")
        else:
            residuals = np.average(residual_array, axis=0)

        # run the sigma_clip algorithm until converged, and we've identified the outliers
        filtered_data = sigma_clip(residuals, sig=sigma, iters=None)
        mask = filtered_data.mask
        wl = self.wl

        sigma0 = config['region_priors']['sigma0']
        logAmp = config["region_params"]["logAmp"]
        sigma = config["region_params"]["sigma"]

        # Sort in decreasing strength of residual
        self.nregions = 0
        regions = {}

        region_mus = {}
        for w, resid in sorted(zip(wl[mask], np.abs(residuals[mask])), key=itemgetter(1), reverse=True):
            if w in wl[covered]:
                continue
            else:
                # check to make sure region is not *right* at the edge of the echelle order
                if w <= np.min(wl) or w >= np.max(wl):
                    continue
                else:
                    # instantiate region and update coverage

                    # Default amp and sigma values
                    regions[self.nregions] = {"logAmp":logAmp, "sigma":sigma, "mu":w}
                    region_mus[self.nregions] = w # for evaluating the mu prior
                    self.nregions += 1

                    # determine the stretch of wl covered by this new region
                    ind = (wl >= (w - sigma0)) & (wl <= (w + sigma0))
                    # update the covered regions
                    covered = covered | ind

        # Take the current nuisance positions as a starting point, and add the regions
        starting_dict = self.sampler.params.copy()
        starting_dict["regions"] = regions

        region_mus = np.array([region_mus[i] for i in range(self.nregions)])

        # Setup the priors
        region_priors = config["region_priors"]
        region_priors.update({"mus":region_mus})
        prior_params = {"regions":region_priors}

        # do all this crap again
        cheb_MH_cov = float(config["cheb_jump"])**2 * np.ones((self.npoly,))
        cov_MH_cov = np.array([float(config["cov_jump"][key]) for key in self.sampler.cov_tup])**2
        region_MH_cov = [float(config["region_jump"][key])**2 for key in C.cov_region_parameters]
        regions_MH_cov = np.array([region_MH_cov for i in range(self.nregions)]).flatten()

        nuisance_MH_cov = np.diag(np.concatenate((cheb_MH_cov, cov_MH_cov, regions_MH_cov)))

        print(starting_dict)
        print("cov shape {}".format(nuisance_MH_cov.shape))

        # Initialize a new sampler, replacing the old one
        self.sampler = NuisanceSampler(OrderModel=self, starting_param_dict=starting_dict, cov=nuisance_MH_cov, debug=True, outdir=self.noutdir, prior_params=prior_params, order=self.order)

        self.p0 = self.sampler.p0

        # Update the nuisance parameters to the starting values so that we at least have a self.data_mat
        print("Updating nuisance parameter data products to starting values.")
        self.update_nuisance(starting_dict)
        self.lnprob = self.evaluate()

        # To speed up convergence, try just doing a bunch of nuisance runs before
        # going into the iteration pattern
        print("Doing nuisance burn-in for {} samples".format(config["nuisance_burn"]))
        self.independent_sample(config["nuisance_burn"])

    def get_lnprob(self, *args):
        '''
        Return the *current* value of lnprob.

        Intended to be called from the master process (StellarSampler.sample), to
        query the child processes for their current value of lnprob.
        '''
        return self.lnprob

    def stellar_lnprob(self, params):
        '''
        Update the model to the parameters and then evaluate the lnprob.

        Intended to be called from the master process via the command "LNPROB".
        '''

        try:
            self.update_stellar(params)
            lnp = self.evaluate() # Also sets self.lnprob to new value
            return lnp
        except C.ModelError:
            self.logger.debug("ModelError in stellar parameters, sending back -np.inf {}".format(params))
            return -np.inf

    def evaluate(self):
        '''
        Return the lnprob using the current version of the DataCovariance matrix
        and other intermediate products.
        '''
        self.lnprob_last = self.lnprob

        X = (self.ChebyshevSpectrum.k * self.flux_std * np.eye(self.npoints)).dot(self.eigenspectra.T)

        CC = X.dot(self.C_GP.dot(X.T)) + self.data_mat

        R = self.fl - self.ChebyshevSpectrum.k * self.flux_mean - X.dot(self.mus)

        try:
            factor, flag = cho_factor(CC)
        except np.linalg.LinAlgError as e:
            self.logger.debug("self.sampler.params are {}".format(self.sampler.params))
            raise C.ModelError("Can't Cholesky factor {}".format(e))

        logdet = np.sum(2 * np.log((np.diag(factor))))

        self.lnprob = -0.5 * (np.dot(R, cho_solve((factor, flag), R)) + logdet) + self.prior

        if self.counter % 100 == 0:
            self.resid_deque.append(R)

        self.counter += 1

        return self.lnprob

    def revert_stellar(self):
        '''
        Revert the status of the model from a rejected stellar proposal.
        '''

        self.logger.debug("Reverting stellar parameters")

        self.lnprob = self.lnprob_last

        self.flux_mean = self.flux_mean_last
        self.flux_std = self.flux_std_last
        self.eigenspectra = self.eigenspectra_last

        self.mus, self.vars = self.mus_last, self.vars_last
        self.C_GP = self.C_GP_last

    def update_stellar(self, params):
        '''
        Update the model to the current stellar parameters.
        '''

        self.logger.debug("Updating stellar parameters to {}".format(params))

        # Store the current accepted values before overwriting with new proposed values.
        self.flux_mean_last = self.flux_mean
        self.flux_std_last = self.flux_std
        self.eigenspectra_last = self.eigenspectra
        self.mus_last, self.vars_last = self.mus, self.vars
        self.C_GP_last = self.C_GP

        #TODO: Possible speedups:
        # 1. Store the EIGENSPECTRA pre-FFT'd

        # Shift the velocity
        vz = params["vz"]
        # Local, shifted copy
        wl_FFT = self.wl_FFT * np.sqrt((C.c_kms + vz) / (C.c_kms - vz))

        # FFT and convolve operations
        vsini = params["vsini"]

        if vsini < 0.2:
            raise C.ModelError("vsini must be positive")

        FF = np.fft.rfft(self.EIGENSPECTRA, axis=1)

        # Determine the stellar broadening kernel
        ub = 2. * np.pi * vsini * self.ss
        sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
        # set zeroth frequency to 1 separately (DC term)
        sb[0] = 1.

        # institute velocity and instrumental taper
        FF_tap = FF * sb

        # do ifft
        eigenspectra_full = np.fft.irfft(FF_tap, len(wl_FFT), axis=1)

        # Spectrum resample operations
        if min(self.wl) < min(wl_FFT) or max(self.wl) > max(wl_FFT):
            raise RuntimeError("Data wl grid ({:.2f},{:.2f}) must fit within the range of wl_FFT ({"
                       ":.2f},{:.2f})".format(min(self.wl), max(self.wl), min(wl_FFT), max(wl_FFT)))

        # Take the output from the FFT operation (eigenspectra_full), and stuff them
        # into respective data products
        for lres, hres in zip(chain([self.flux_mean, self.flux_std], self.eigenspectra), eigenspectra_full):
            interp = InterpolatedUnivariateSpline(wl_FFT, hres, k=5)
            lres[:] = interp(self.wl)
            del interp

        gc.collect()

        # Adjust flux_mean and flux_std by Omega
        Omega = 10**params["logOmega"]
        self.flux_mean *= Omega
        self.flux_std *= Omega

        # Now update the parameters from the emulator
        pars = np.array([params["temp"], params["logg"], params["Z"]])

        # If pars are outside the grid, Emulator will raise C.ModelError
        self.mus, self.vars = self.Emulator(pars)

        self.C_GP = self.vars * np.eye(self.ncomp)

    def decide_stellar(self, yes):
        '''
        Interpret the decision from the master process to either revert the
        stellar model (rejected parameters) or move on (accepted parameters).
        '''
        if yes:
            # accept and move on
            self.logger.debug("Deciding to accept stellar parameters")
        else:
            # revert and move on
            self.logger.debug("Deciding to revert stellar parameters")
            self.revert_stellar()

        # Proceed with independent sampling
        self.independent_sample(1)

    def update_nuisance(self, params):
        '''
        Update the nuisance parameters and data covariance matrix.

        :param params: large dictionary containing cheb, cov, and regions
        '''

        self.logger.debug("Updating nuisance parameters to {}".format(params))
        # Read off the Chebyshev parameters and update
        self.ChebyshevSpectrum.update(params["cheb"])

        # Create the full data covariance matrix.
        l = params["cov"]["l"]
        sigAmp = params["cov"]["sigAmp"]

        # Check to make sure the global covariance parameters make sense
        if sigAmp < 0.1:
            raise C.ModelError("sigAmp shouldn't be lower than 0.1, something is wrong.")

        max_r = 6.0 * l # [km/s]

        # Check all regions, take the max
        if self.nregions > 0:
            regions = params["regions"]
            keys = sorted(regions)
            sigmas = np.array([regions[key]["sigma"] for key in keys]) #km/s
            #mus = np.array([regions[key]["mu"] for key in keys])
            max_reg = 4.0 * np.max(sigmas)
            #If this is a larger distance than the global length, replace it
            max_r = max_reg if max_reg > max_r else max_r
            #print("Max_r now set by regions {}".format(max_r))

        # print("max_r is {}".format(max_r))

        # Create a partial function which returns the proper element.
        k_func = make_k_func(params)

        # Store the previous data matrix in case we want to revert later
        self.data_mat_last = self.data_mat
        self.data_mat = get_dense_C(self.wl, k_func=k_func, max_r=max_r) + sigAmp*self.sigma_matrix

    def revert_nuisance(self, *args):
        '''
        Revert all products from the nuisance parameters, including the data
        covariance matrix.
        '''

        self.logger.debug("Reverting nuisance parameters")

        self.lnprob = self.lnprob_last

        self.ChebyshevSpectrum.revert()
        self.data_mat = self.data_mat_last

    def clear_resid_deque(self):
        '''
        Clear the accumulated residual spectra.
        '''
        self.resid_deque.clear()

    def independent_sample(self, niter):
        '''
        Do the independent sampling specific to this echelle order, using the
        attached self.sampler (NuisanceSampler).

        :param niter: number of iterations to complete before returning to master process.

        '''

        self.logger.debug("Beginning independent sampling on nuisance parameters")

        if self.lnprob:
            # If we have a current value, pass it to the sampler
            self.p0, self.lnprob, state = self.sampler.run_mcmc(pos0=self.p0, N=niter, lnprob0=self.lnprob)
        else:
            # Otherwise, start from the beginning
            self.p0, self.lnprob, state = self.sampler.run_mcmc(pos0=self.p0, N=niter)

        self.logger.debug("Finished independent sampling on nuisance parameters")
        # Don't return anything to the master process.

    def finish(self, *args):
        '''
        Wrap up the sampling and write the samples to disk.
        '''

        print(self.sampler.acceptance_fraction)
        print(self.sampler.acor)
        self.sampler.write()
        self.sampler.plot() # triangle_plot=True
        print("There were {} exceptions.".format(len(self.exceptions)))
        # print out the values of each region key.
        for exception in self.exceptions:
            regions = exception["regions"]
            keys = sorted(regions)
            for key in keys:
                print(regions[key])
            cov = exception["cov"]
            print(cov)
            print("\n\n")

    def brain(self, conn):
        '''
        The infinite loop of the subprocess, which continues to listen for
        messages on the pipe.
        '''
        self.conn = conn
        alive = True
        while alive:
            #Keep listening for messages put on the Pipe
            alive = self.interpret()
            #Once self.interpret() returns `False`, this loop will die.
        self.conn.send("DEAD")

    def interpret(self):
        '''
        Interpret the messages being put into the Pipe, and do something with
        them. Messages are always sent in a 2-arg tuple (fname, arg)
        Right now we only expect one function and one argument but this could
        be generalized to **args.
        '''
        #info("brain")

        fname, arg = self.conn.recv() # Waits here to receive a new message
        self.logger.debug("{} received message {}".format(os.getpid(), (fname, arg)))

        func = self.func_dict.get(fname, False)
        if func:
            response = func(arg)
        else:
            self.logger.info("Given an unknown function {}, assuming kill signal.".format(fname))
            return False

        # Functions only return a response other than None when they want them
        # communicated back to the master process.
        # Some commands sent to the child processes do not require a response
        # to the main process.
        if response:
            self.logger.debug("{} sending back {}".format(os.getpid(), response))
            self.conn.send(response)
        return True

# We create one OrderModel in the main process. When the process forks, each
# subprocess now has its own independent OrderModel instance.
# Then, each forked model will be customized using an INIT command passed
# through the PIPE.

model = OrderModel(debug=True)

# Comment out these following lines to profile
# Fork a subprocess for each key: (spectra, order)
pconns = {} # Parent connections
cconns = {} # Child connections
ps = {}
for spectrum_key in spectra_keys:
    for order_key in order_keys:
        pconn, cconn = Pipe()
        key = (spectrum_key, order_key)
        pconns[key], cconns[key] = pconn, cconn
        p = Process(target=model.brain, args=(cconn,))
        p.start()
        ps[key] = p

# Initialize all of the orders to a specific DataSpectrum and echelle order
for key, pconn in pconns.items():
    pconn.send(("INIT", key))

# From here on, this script operates on the master process only.
if args.perturb:
    perturb(stellar_Starting, config["stellar_jump"], factor=args.perturb)

def profile_code():
    '''
    Test hook designed to be used by cprofile or kernprof. Does not include any
    network latency from communicating or synchronizing between processes
    because we run on just one process.
    '''

    #Evaluate one complete iteration from delivery of stellar parameters from master process

    #Master proposal
    stellar_Starting.update({"logg":4.29})
    model.stellar_lnprob(stellar_Starting)
    #Assume we accepted
    model.decide_stellar(True)

    #Right now, assumes Kurucz order 23


def main():

    # Uncomment these lines to profile
    # #Initialize the current model for profiling purposes
    # model.initialize((0, 0))
    # import cProfile
    # cProfile.run("profile_code()", "prof")
    # import sys; sys.exit()

    mySampler = StellarSampler(pconns=pconns, starting_param_dict=stellar_Starting,
        cov=stellar_MH_cov, outdir=outdir, debug=True, fix_logg=fix_logg)

    mySampler.run_mcmc(mySampler.p0, config['burn_in'])
    #mySampler.reset()

    self.logger.info("Instantiating Regions")

    # Now that we are burned in, instantiate any regions
    for key, pconn in pconns.items():
        pconn.send(("INST", None))

    mySampler.run_mcmc(mySampler.p0, config['samples'])

    print(mySampler.acceptance_fraction)
    print(mySampler.acor)
    mySampler.write()
    mySampler.plot() #triangle_plot = True

    # Kill all of the orders
    for pconn in pconns.values():
        pconn.send(("FINISH", None))
        pconn.send(("DIE", None))

    # Join on everything and terminate
    for p in ps.values():
        p.join()
        p.terminate()

    import sys;sys.exit()

if __name__=="__main__":
    main()

# All subprocesses will inherit pipe file descriptors created in the master process.
# http://www.pushingbits.net/posts/python-multiprocessing-with-pipes/
# thus, to really close a pipe, you need to close it in every subprocess.
