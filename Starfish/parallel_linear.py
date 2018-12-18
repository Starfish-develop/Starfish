# Parallel implementation for sampling a multi-order echelle spectrum.
# Because the likelihood calculation is independent for each order, the
# runtime is essentially constant regardless of how large a spectral range is used.

# Additionally, one could use this to fit multiple stars at once.

# parallel.py is meant to be run by other modules that import it and use the objects.
# It has an argparser because I think it's the easiest way to consolidate all of the
# parameters into one place, but I'm open to new suggestions.

import argparse
parser = argparse.ArgumentParser(prog="parallel.py", description="Run Starfish fitting model in parallel.")
parser.add_argument("-r", "--run_index", help="All data will be written into this directory, overwriting any that exists. Default is current working directory.")
# Even though these arguments aren't being used, we need to add them.
parser.add_argument("--generate", action="store_true", help="Write out the data, mean model, and residuals for each order.")
parser.add_argument("--initPhi", action="store_true", help="Create *phi.json files for each order using values in config.yaml")
parser.add_argument("--optimize", choices=["Theta", "Phi", "Cheb"], help="Optimize the Theta or Phi parameters, keeping the alternate set of parameters fixed.")
parser.add_argument("--sample", choices=["ThetaCheb", "ThetaPhi", "ThetaPhiLines"], help="Sample the parameters, keeping the alternate set of parameters fixed.")
parser.add_argument("--samples", type=int, default=5, help="How many samples to run?")
args = parser.parse_args()

from multiprocessing import Process, Pipe
import os
import numpy as np

from Starfish import config
import Starfish.grid_tools
from Starfish.grid_tools import Interpolator, HDF5Interface
from Starfish.samplers import StateSampler
from Starfish.spectrum import DataSpectrum, Mask, ChebyshevSpectrum

import Starfish.constants as C
from Starfish.covariance import get_dense_C, make_k_func, make_k_func_region
from Starfish.model import ThetaParam, PhiParam

from scipy.special import j1
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet
from astropy.stats import sigma_clip

import gc
import logging

from itertools import chain
from collections import deque
from operator import itemgetter
import yaml
import shutil
import json

def init_directories(run_index=None):
    '''
    If we are sampling, then we need to setup output directories to store the
    samples and other output products. If we're sampling, we probably want to be
    running multiple chains at once, and so we have to set things up so that they
    don't conflict.

    :returns: routdir, the outdir for this current run.
    '''

    base = config.outdir + config.name + "/run{:0>2}/"

    if run_index is None:
        run_index = 0
        while os.path.exists(base.format(run_index)):
            print(base.format(run_index), "exists")
            run_index += 1
        routdir = base.format(run_index)

    else:
        routdir = base.format(run_index)
        #Delete this routdir, if it exists
        if os.path.exists(routdir):
            print("Deleting", routdir)
            shutil.rmtree(routdir)

    print("Creating ", routdir)
    os.makedirs(routdir)

    # Copy yaml file to routdir for archiving purposes
    shutil.copy("config.yaml", routdir + "config.yaml")

    # Create subdirectories
    for model_number in range(len(config.data["files"])):
        for order in config.data["orders"]:
            order_dir = routdir + config.specfmt.format(model_number, order)
            print("Creating ", order_dir)
            os.makedirs(order_dir)

    return routdir

if args.run_index:
    config.routdir = init_directories(args.run_index)
else:
    config.routdir = ""

# list of keys from 0 to (norders - 1)
order_keys = np.arange(len(config.data["orders"]))
DataSpectra = [DataSpectrum.open(file, orders=config.data["orders"]) for file in config.data["files"]]
# list of keys from 0 to (nspectra - 1) Used for indexing purposes.
spectra_keys = np.arange(len(DataSpectra))

#Instruments are provided as one per dataset
Instruments = [eval("Starfish.grid_tools." + inst)() for inst in config.data["instruments"]]

masks = config.get("mask", None)
if masks is not None:
    for mask, dataSpec in zip(masks, DataSpectra):
        myMask = Mask(mask, orders=config.data["orders"])
        dataSpec.add_mask(myMask.masks)

# Set up the logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", filename="{}log.log".format(
    config.routdir), level=logging.DEBUG, filemode="w", datefmt='%m/%d/%Y %I:%M:%S %p')
#
# def perturb(startingDict, jumpDict, factor=3.):
#     '''
#     Given a starting parameter dictionary loaded from a config file, perturb the
#     values as a multiple of the jump distribution. This is designed so that
#     not all chains start at exactly the same place.
#
#     Modifies the startingDict
#     '''
#     for key in startingDict.keys():
#         startingDict[key] += factor * np.random.normal(loc=0, scale=jumpDict[key])
#

# fix_logg = config.get("fix_logg", None)

# Updating specific covariances to speed mixing
if config.get("use_cov", None):
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


class Order:
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
                          "DECIDE": self.decide_Theta,
                          "INST": self.instantiate,
                          "LNPROB": self.lnprob_Theta,
                          "GET_LNPROB": self.get_lnprob,
                          "FINISH": self.finish,
                          "SAVE": self.save,
                          "OPTIMIZE_CHEB": self.optimize_Cheb
                          }

        self.debug = debug
        self.logger = logging.getLogger("{}".format(self.__class__.__name__))

    def initialize(self, key):
        '''
        Initialize to the correct chunk of data (echelle order).

        :param key: (spectrum_id, order_key)
        :param type: (int, int)

        This method should only be called after all subprocess have been forked.
        '''

        self.id = key
        spectrum_id, self.order_key = self.id
        # Make sure these are ints
        self.spectrum_id = int(spectrum_id)

        self.instrument = Instruments[self.spectrum_id]
        self.dataSpectrum = DataSpectra[self.spectrum_id]
        self.wl = self.dataSpectrum.wls[self.order_key]
        self.fl = self.dataSpectrum.fls[self.order_key]
        self.sigma = self.dataSpectrum.sigmas[self.order_key]
        self.ndata = len(self.wl)
        self.mask = self.dataSpectrum.masks[self.order_key]
        self.order = int(self.dataSpectrum.orders[self.order_key])

        self.logger = logging.getLogger("{} {}".format(self.__class__.__name__, self.order))
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.logger.info("Initializing model on Spectrum {}, order {}.".format(self.spectrum_id, self.order_key))

        self.npoly = config["cheb_degree"]
        self.chebyshevSpectrum = ChebyshevSpectrum(self.dataSpectrum, self.order_key, npoly=self.npoly)

        # If the file exists, optionally initiliaze to the chebyshev values
        fname = config.specfmt.format(self.spectrum_id, self.order) + "phi.json"
        if os.path.exists(fname):
            self.logger.debug("Loading stored Chebyshev parameters.")
            phi = PhiParam.load(fname)
            self.chebyshevSpectrum.update(phi.cheb)

        self.resid_deque = deque(maxlen=500) #Deque that stores the last residual spectra, for averaging
        self.counter = 0

        self.interpolator = Interpolator(self.wl, HDF5Interface())
        self.flux = None # Where the interpolator will store the flux

        self.wl_FFT = self.interpolator.wl

        # The raw eigenspectra and mean flux components

        self.ss = np.fft.rfftfreq(len(self.wl_FFT), d=self.interpolator.interface.dv)
        self.ss[0] = 0.01 # junk so we don't get a divide by zero error

        self.sigma_mat = self.sigma**2 * np.eye(self.ndata)

        self.lnprior = 0.0 # Modified and set by NuisanceSampler.lnprob

        # self.nregions = 0
        # self.exceptions = []

        # Update the outdir based upon id
        self.noutdir = config.routdir + "{}/{}/".format(self.spectrum_id, self.order)

    def instantiate(self, *args):
        '''
        If mixing Theta and Phi optimization/sampling, perform the sigma clipping
        operation to instantiate covariant regions to cover outliers.

        May involve creating a new NuisanceSampler.
        '''
        raise NotImplementedError

    def get_lnprob(self, *args):
        '''
        Return the *current* value of lnprob.

        Intended to be called from the master process to
        query the child processes for their current value of lnprob.
        '''
        return self.lnprob

    def lnprob_Theta(self, p):
        '''
        Update the model to the Theta parameters and then evaluate the lnprob.

        Intended to be called from the master process via the command "LNPROB".
        '''
        try:
            self.update_Theta(p)
            lnp = self.evaluate() # Also sets self.lnprob to new value
            return lnp
        except (C.ModelError, C.InterpolationError):
            self.logger.debug("ModelError in stellar parameters, sending back -np.inf {}".format(p))
            return -np.inf

    def evaluate(self):
        '''
        Return the lnprob using the current version of the C_GP matrix, data matrix,
        and other intermediate products.
        '''

        self.lnprob_last = self.lnprob

        CC = self.data_mat

        model = self.chebyshevSpectrum.k * self.flux

        try:

            factor, flag = cho_factor(CC)

            R = self.fl - model

            logdet = np.sum(2 * np.log((np.diag(factor))))
            self.lnprob = -0.5 * (np.dot(R, cho_solve((factor, flag), R)) + logdet)

            self.logger.debug("Evaluating lnprob={}".format(self.lnprob))
            return self.lnprob

        # To give us some debugging information about what went wrong.
        except np.linalg.linalg.LinAlgError:
            print("Spectrum:", self.spectrum_id, "Order:", self.order)
            raise

    def update_Theta(self, p):
        '''
        Update the model to the current Theta parameters.

        :param p: parameters to update model to
        :type p: model.ThetaParam
        '''

        # Dirty hack
        fix_logg = config.get("fix_logg", None)
        if fix_logg is not None:
            p.grid[1] = fix_logg
        print("grid pars are", p.grid)

        self.logger.debug("Updating Theta parameters to {}".format(p))

        # Store the current accepted values before overwriting with new proposed values.
        self.flux_last = self.flux

        # Local, shifted copy of wavelengths
        wl_FFT = self.wl_FFT * np.sqrt((C.c_kms + p.vz) / (C.c_kms - p.vz))

        flux_raw = self.interpolator(p.grid)

        # If vsini is less than 0.2 km/s, we might run into issues with
        # the grid spacing. Therefore skip the convolution step if we have
        # values smaller than this.
        # FFT and convolve operations
        if p.vsini < 0.0:
            raise C.ModelError("vsini must be positive")
        elif p.vsini < 0.2:
            # Skip the vsini taper due to instrumental effects
            flux_taper = flux_raw
        else:
            FF = np.fft.rfft(flux_raw)

            # Determine the stellar broadening kernel
            ub = 2. * np.pi * p.vsini * self.ss
            sb = j1(ub) / ub - 3 * np.cos(ub) / (2 * ub ** 2) + 3. * np.sin(ub) / (2 * ub ** 3)
            # set zeroth frequency to 1 separately (DC term)
            sb[0] = 1.

            # institute vsini taper
            FF_tap = FF * sb

            # do ifft
            flux_taper = np.fft.irfft(FF_tap, len(self.wl_FFT))

        # Spectrum resample operations
        if min(self.wl) < min(wl_FFT) or max(self.wl) > max(wl_FFT):
            raise RuntimeError("Data wl grid ({:.2f},{:.2f}) must fit within the range of wl_FFT ({:.2f},{:.2f})".format(min(self.wl), max(self.wl), min(wl_FFT), max(wl_FFT)))

        # Take the output from the FFT operation and stuff it into the respective data products
        interp = InterpolatedUnivariateSpline(wl_FFT, flux_taper, k=5)
        self.flux = interp(self.wl)
        del interp

        gc.collect()

        # Adjust flux_mean and flux_std by Omega
        Omega = 10**p.logOmega
        self.flux *= Omega

    def revert_Theta(self):
        '''
        Revert the status of the model from a rejected Theta proposal.
        '''

        self.logger.debug("Reverting Theta parameters")

        self.lnprob = self.lnprob_last

        self.flux = self.flux_last

    def decide_Theta(self, yes):
        '''
        Interpret the decision from the master process to either revert the
        Theta model (rejected parameters) or move on (accepted parameters).

        :param yes: if True, accept stellar parameters.
        :type yes: boolean
        '''
        if yes:
            # accept and move on
            self.logger.debug("Deciding to accept Theta parameters")
        else:
            # revert and move on
            self.logger.debug("Deciding to revert Theta parameters")
            self.revert_Theta()

        # Proceed with independent sampling
        self.independent_sample(1)

    def optimize_Cheb(self, *args):
        '''
        Keeping the current Theta parameters fixed and assuming white noise,
        optimize the Chebyshev parameters
        '''

        # self.fix_c0 = True if index == (len(DataSpectrum.wls) - 1) else False #Fix the last c0
        # This is necessary if we want to update just a single order.

        if self.chebyshevSpectrum.fix_c0 & len(self.dataSpectrum.wls) > 1:
            p0 = np.zeros((self.npoly - 1))
        else:
            self.chebyshevSpectrum.fix_c0 = False
            p0 = np.zeros((self.npoly))

        def fprob(p):
            self.chebyshevSpectrum.update(p)
            lnp = self.evaluate()
            print(self.order, p, lnp)
            if lnp == -np.inf:
                return 1e99
            else:
                return -lnp

        from scipy.optimize import fmin
        result = fmin(fprob, p0, maxiter=10000, maxfun=10000)
        print(self.order, result)

        # Due to a JSON bug, np.int64 type objects will get read twice,
        # and cause this routine to fail. Therefore we have to be careful
        # to convert these to ints.
        phi = PhiParam(spectrum_id=int(self.spectrum_id), order=int(self.order), fix_c0=self.chebyshevSpectrum.fix_c0, cheb=result)
        phi.save()

    def update_Phi(self, p):
        '''
        Update the Phi parameters and data covariance matrix.

        :param params: large dictionary containing cheb, cov, and regions
        '''

        raise NotImplementedError

    def revert_Phi(self, *args):
        '''
        Revert all products from the nuisance parameters, including the data
        covariance matrix.
        '''

        self.logger.debug("Reverting Phi parameters")

        self.lnprob = self.lnprob_last

        self.chebyshevSpectrum.revert()
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

        self.logger.debug("Beginning independent sampling on Phi parameters")

        if self.lnprob:
            # If we have a current value, pass it to the sampler
            self.p0, self.lnprob, state = self.sampler.run_mcmc(pos0=self.p0, N=niter, lnprob0=self.lnprob)
        else:
            # Otherwise, start from the beginning
            self.p0, self.lnprob, state = self.sampler.run_mcmc(pos0=self.p0, N=niter)

        self.logger.debug("Finished independent sampling on Phi parameters")
        # Don't return anything to the master process.

    def finish(self, *args):
        '''
        Wrap up the sampling and write the samples to disk.
        '''
        self.logger.debug("Finishing")

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

    def save(self, *args):
        '''
        Using the current values for flux, write out the data, mean model, and mean
        residuals into a JSON.
        '''

        resid = self.fl - self.flux

        my_dict = {"wl":self.wl.tolist(), "data":self.fl.tolist(), "model":self.flux.tolist(), "resid":resid.tolist(), "sigma":self.sigma.tolist(), "spectrum_id":self.spectrum_id, "order":self.order}

        fname = config.specfmt.format(self.spectrum_id, self.order)
        f = open(fname + "spec.json", 'w')
        json.dump(my_dict, f, indent=2, sort_keys=True)
        f.close()


class OptimizeTheta(Order):

    def initialize(self, key):
        super().initialize(key)
        # Any additional setup here

        # for now, just use white noise
        self.data_mat = self.sigma_mat.copy()


class OptimizeCheb(Order):
    def initialize(self, key):
        super().initialize(key)
        # Any additional setup here

        # for now, just use white noise
        self.data_mat = self.sigma_mat.copy()


class OptimizePhi(Order):
    def __init__(self):
        pass

class SampleThetaCheb(Order):
    def initialize(self, key):
        super().initialize(key)

        # for now, just use white noise
        self.data_mat = self.sigma_mat.copy()
        self.data_mat_last = self.data_mat.copy()

        #Set up p0 and the independent sampler
        fname = config.specfmt.format(self.spectrum_id, self.order) + "phi.json"
        phi = PhiParam.load(fname)
        self.p0 = phi.cheb
        cov = np.diag(config["cheb_jump"]**2 * np.ones(len(self.p0)))

        def lnfunc(p):
            # turn this into pars
            self.update_Phi(p)
            lnp = self.evaluate()
            self.logger.debug("Evaluated Phi parameters: {} {}".format(p, lnp))
            return lnp

        def rejectfn():
            self.logger.debug("Calling Phi revertfn.")
            self.revert_Phi()

        self.sampler = StateSampler(lnfunc, self.p0, cov, query_lnprob=self.get_lnprob, rejectfn=rejectfn, debug=True)

    def update_Phi(self, p):
        '''
        Update the Chebyshev coefficients only.
        '''
        self.chebyshevSpectrum.update(p)

    def finish(self, *args):
        super().finish(*args)
        fname = config.routdir + config.specfmt.format(self.spectrum_id, self.order) + "/mc.hdf5"
        self.sampler.write(fname=fname)

class SampleThetaPhi(Order):

    def initialize(self, key):
        # Run through the standard initialization
        super().initialize(key)

        # for now, start with white noise
        self.data_mat = self.sigma_mat.copy()
        self.data_mat_last = self.data_mat.copy()

        #Set up p0 and the independent sampler
        fname = config.specfmt.format(self.spectrum_id, self.order) + "phi.json"
        phi = PhiParam.load(fname)

        # Set the regions to None, since we don't want to include them even if they
        # are there
        phi.regions = None

        #Loading file that was previously output
        # Convert PhiParam object to an array
        self.p0 = phi.toarray()

        jump = config["Phi_jump"]
        cheb_len = (self.npoly - 1) if self.chebyshevSpectrum.fix_c0 else self.npoly
        cov_arr = np.concatenate((config["cheb_jump"]**2 * np.ones((cheb_len,)), np.array([jump["sigAmp"], jump["logAmp"], jump["l"]])**2 ))
        cov = np.diag(cov_arr)

        def lnfunc(p):
            # Convert p array into a PhiParam object
            ind = self.npoly
            if self.chebyshevSpectrum.fix_c0:
                ind -= 1

            cheb = p[0:ind]
            sigAmp = p[ind]
            ind+=1
            logAmp = p[ind]
            ind+=1
            l = p[ind]

            par = PhiParam(self.spectrum_id, self.order, self.chebyshevSpectrum.fix_c0, cheb, sigAmp, logAmp, l)

            self.update_Phi(par)
            lnp = self.evaluate()
            self.logger.debug("Evaluated Phi parameters: {} {}".format(par, lnp))
            return lnp

        def rejectfn():
            self.logger.debug("Calling Phi revertfn.")
            self.revert_Phi()

        self.sampler = StateSampler(lnfunc, self.p0, cov, query_lnprob=self.get_lnprob, rejectfn=rejectfn, debug=True)

    def update_Phi(self, p):
        self.logger.debug("Updating nuisance parameters to {}".format(p))

        # Read off the Chebyshev parameters and update
        self.chebyshevSpectrum.update(p.cheb)

        # Check to make sure the global covariance parameters make sense
        if p.sigAmp < 0.1:
            raise C.ModelError("sigAmp shouldn't be lower than 0.1, something is wrong.")

        max_r = 6.0 * p.l # [km/s]

        # Create a partial function which returns the proper element.
        k_func = make_k_func(p)

        # Store the previous data matrix in case we want to revert later
        self.data_mat_last = self.data_mat
        self.data_mat = get_dense_C(self.wl, k_func=k_func, max_r=max_r) + p.sigAmp*self.sigma_mat

    def finish(self, *args):
        super().finish(*args)
        fname = config.routdir + config.specfmt.format(self.spectrum_id, self.order) + "/mc.hdf5"
        self.sampler.write(fname=fname)

class SampleThetaPhiLines(Order):

    def initialize(self, key):
        # Run through the standard initialization
        super().initialize(key)

        # for now, start with white noise
        self.data_mat = self.sigma_mat.copy()
        self.data_mat_last = self.data_mat.copy()

        #Set up p0 and the independent sampler
        fname = config.specfmt.format(self.spectrum_id, self.order) + "phi.json"
        phi = PhiParam.load(fname)

        # print("Phi.regions", phi.regions)
        # import sys
        # sys.exit()
        # Get the regions matrix
        region_func = make_k_func_region(phi)

        max_r = 4.0 * np.max(phi.regions, axis=0)[2]

        self.region_mat = get_dense_C(self.wl, k_func=region_func, max_r=max_r)

        print(self.region_mat)

        # Then set phi to None
        phi.regions = None

        #Loading file that was previously output
        # Convert PhiParam object to an array
        self.p0 = phi.toarray()

        jump = config["Phi_jump"]
        cheb_len = (self.npoly - 1) if self.chebyshevSpectrum.fix_c0 else self.npoly
        cov_arr = np.concatenate((config["cheb_jump"]**2 * np.ones((cheb_len,)), np.array([jump["sigAmp"], jump["logAmp"], jump["l"]])**2 ))
        cov = np.diag(cov_arr)

        def lnfunc(p):
            # Convert p array into a PhiParam object
            ind = self.npoly
            if self.chebyshevSpectrum.fix_c0:
                ind -= 1

            cheb = p[0:ind]
            sigAmp = p[ind]
            ind+=1
            logAmp = p[ind]
            ind+=1
            l = p[ind]

            phi = PhiParam(self.spectrum_id, self.order, self.chebyshevSpectrum.fix_c0, cheb, sigAmp, logAmp, l)

            self.update_Phi(phi)
            lnp = self.evaluate()
            self.logger.debug("Evaluated Phi parameters: {} {}".format(phi, lnp))
            return lnp

        def rejectfn():
            self.logger.debug("Calling Phi revertfn.")
            self.revert_Phi()

        self.sampler = StateSampler(lnfunc, self.p0, cov, query_lnprob=self.get_lnprob, rejectfn=rejectfn, debug=True)

    def update_Phi(self, phi):
        self.logger.debug("Updating nuisance parameters to {}".format(phi))

        # Read off the Chebyshev parameters and update
        self.chebyshevSpectrum.update(phi.cheb)

        # Check to make sure the global covariance parameters make sense
        if phi.sigAmp < 0.1:
            raise C.ModelError("sigAmp shouldn't be lower than 0.1, something is wrong.")

        max_r = 6.0 * phi.l # [km/s]

        # Create a partial function which returns the proper element.
        k_func = make_k_func(phi)

        # Store the previous data matrix in case we want to revert later
        self.data_mat_last = self.data_mat
        self.data_mat = get_dense_C(self.wl, k_func=k_func, max_r=max_r) + phi.sigAmp*self.sigma_mat + self.region_mat

    def finish(self, *args):
        super().finish(*args)
        fname = config.routdir + config.specfmt.format(self.spectrum_id, self.order) + "/mc.hdf5"
        self.sampler.write(fname=fname)


# class SampleThetaPhiLines(Order):
#     def instantiate(self, *args):
#         # threshold for sigma clipping
#         sigma=config["sigma_clip"]
#
#         # array that specifies if a pixel is already covered.
#         # to start, it should be all False
#         covered = np.zeros((self.ndata,), dtype='bool')
#
#         #average all of the spectra in the deque together
#         residual_array = np.array(self.resid_deque)
#         if len(self.resid_deque) == 0:
#             raise RuntimeError("No residual spectra stored yet.")
#         else:
#             residuals = np.average(residual_array, axis=0)
#
#         # run the sigma_clip algorithm until converged, and we've identified the outliers
#         filtered_data = sigma_clip(residuals, sig=sigma, iters=None)
#         mask = filtered_data.mask
#         wl = self.wl
#
#         sigma0 = config['region_priors']['sigma0']
#         logAmp = config["region_params"]["logAmp"]
#         sigma = config["region_params"]["sigma"]
#
#         # Sort in decreasing strength of residual
#         self.nregions = 0
#         regions = {}
#
#         region_mus = {}
#         for w, resid in sorted(zip(wl[mask], np.abs(residuals[mask])), key=itemgetter(1), reverse=True):
#             if w in wl[covered]:
#                 continue
#             else:
#                 # check to make sure region is not *right* at the edge of the echelle order
#                 if w <= np.min(wl) or w >= np.max(wl):
#                     continue
#                 else:
#                     # instantiate region and update coverage
#
#                     # Default amp and sigma values
#                     regions[self.nregions] = {"logAmp":logAmp, "sigma":sigma, "mu":w}
#                     region_mus[self.nregions] = w # for evaluating the mu prior
#                     self.nregions += 1
#
#                     # determine the stretch of wl covered by this new region
#                     ind = (wl >= (w - sigma0)) & (wl <= (w + sigma0))
#                     # update the covered regions
#                     covered = covered | ind
#
#         # Take the current nuisance positions as a starting point, and add the regions
#         starting_dict = self.sampler.params.copy()
#         starting_dict["regions"] = regions
#
#         region_mus = np.array([region_mus[i] for i in range(self.nregions)])
#
#         # Setup the priors
#         region_priors = config["region_priors"]
#         region_priors.update({"mus":region_mus})
#         prior_params = {"regions":region_priors}
#
#         # do all this crap again
#         cheb_MH_cov = float(config["cheb_jump"])**2 * np.ones((self.npoly,))
#         cov_MH_cov = np.array([float(config["cov_jump"][key]) for key in self.sampler.cov_tup])**2
#         region_MH_cov = [float(config["region_jump"][key])**2 for key in C.cov_region_parameters]
#         regions_MH_cov = np.array([region_MH_cov for i in range(self.nregions)]).flatten()
#
#         nuisance_MH_cov = np.diag(np.concatenate((cheb_MH_cov, cov_MH_cov, regions_MH_cov)))
#
#         print(starting_dict)
#         print("cov shape {}".format(nuisance_MH_cov.shape))
#
#         # Initialize a new sampler, replacing the old one
#         self.sampler = NuisanceSampler(OrderModel=self, starting_param_dict=starting_dict, cov=nuisance_MH_cov, debug=True, outdir=self.noutdir, prior_params=prior_params, order=self.order)
#
#         self.p0 = self.sampler.p0
#
#         # Update the nuisance parameters to the starting values so that we at least have a self.data_mat
#         print("Updating nuisance parameter data products to starting values.")
#         self.update_nuisance(starting_dict)
#         self.lnprob = self.evaluate()
#
#         # To speed up convergence, try just doing a bunch of nuisance runs before
#         # going into the iteration pattern
#         print("Doing nuisance burn-in for {} samples".format(config["nuisance_burn"]))
#         self.independent_sample(config["nuisance_burn"])

# We create one Order() in the main process. When the process forks, each
# subprocess now has its own independent OrderModel instance.
# Then, each forked model will be customized using an INIT command passed
# through the PIPE.

def initialize(model):
    # Fork a subprocess for each key: (spectra, order)
    pconns = {} # Parent connections
    cconns = {} # Child connections
    ps = {} # Process objects
    # Create all of the pipes
    for spectrum_key in spectra_keys:
        for order_key in order_keys:
            pconn, cconn = Pipe()
            key = (spectrum_key, order_key)
            pconns[key], cconns[key] = pconn, cconn
            p = Process(target=model.brain, args=(cconn,))
            p.start()
            ps[key] = p

    # initialize each Model to a specific DataSpectrum and echelle order
    for key, pconn in pconns.items():
        pconn.send(("INIT", key))

    return (pconns, cconns, ps)


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
