import numpy as np
import sys
import emcee
from emcee import GibbsSampler, GibbsSubController, GibbsController
from . import constants as C
from .grid_tools import Interpolator, ErrorInterpolator
from .spectrum import ModelSpectrum, ChebyshevSpectrum, ModelSpectrumHA
from .covariance import CovarianceMatrix #from StellarSpectra.spectrum import CovarianceMatrix #pure Python
import time
import json
import h5py
from astropy.stats.funcs import sigma_clip
import logging
import os
from collections import deque
from operator import itemgetter
import matplotlib.pyplot as plt
from itertools import zip_longest

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def plot_walkers(filename, samples, labels=None):
    ndim = len(samples[0, :])
    fig, ax = plt.subplots(nrows=ndim, sharex=True)
    for i in range(ndim):
        ax[i].plot(samples[:,i])
        if labels is not None:
            ax[i].set_ylabel(labels[i])
    ax[-1].set_xlabel("Sample number")
    fig.savefig(filename)
    plt.close(fig)

class ModelEncoder(json.JSONEncoder):
    '''
    Designed to serialize an instance of o=Model() to JSON
    '''
    def default(self, o):
        try:
            #We turn Model into a hierarchical dictionary, which will serialize to JSON

            mydict = {"stellar_tuple":o.stellar_tuple, "cheb_tuple": o.cheb_tuple, "cov_tuple": o.cov_tuple,
            "region_tuple": o.region_tuple, "stellar_params": o.stellar_params, "orders": {}}

            #Determine the list of orders
            orders = o.DataSpectrum.orders

            #for each order, then instantiate an order dictionary
            for i,order in enumerate(orders):
                #Will eventually be mydict['orders'] = {"22":order_dict, "23:order_dict, ...}
                order_dict = {}
                order_model = o.OrderModels[i]
                order_dict["cheb"] = order_model.cheb_params
                order_dict["global_cov"] = order_model.global_cov_params

                #Now determine if we need to add any regions
                order_dict["regions"] = order_model.get_regions_dict()

                mydict['orders'].update({str(order): order_dict})

        except TypeError:
            pass
        else:
            return mydict
        # Let the base class default method raise the TypeError, if there is one
        return json.JSONEncoder.default(self, o)

class Model:
    '''
    Container class to create and bring together all of the relevant data and models to aid in evaulation.

    :param DataSpectrum: the data to fit
    :type DataSpectrum: :obj:`spectrum.DataSpectrum` object
    :param Instrument: the instrument with which the data was acquired
    :type Instrument: :obj:`grid_tools.Instrument` object
    :param HDF5Interface: the interface to the synthetic stellar library
    :type HDF5Interface: :obj:`grid_tools.HDF5Interface` object
    :param stellar_tuple: describes the order of parameters. If ``alpha`` is missing, :obj:``grid_tools.Interpolator`` is trilinear.
    :type stellar_tuple: tuple

    '''

    @classmethod
    def from_json(cls, filename, DataSpectrum, Instrument, HDF5Interface, ErrorHDF5Interface):
        '''
        Instantiate from a JSON file.
        '''

        #Determine tuples from the JSON output
        f = open(filename, "r")
        read = json.load(f)
        f.close()

        #Read DataSpectrum, Instrument, HDF5Interface, stellar_tuple, cov_tuple, and region_tuple
        stellar_tuple = tuple(read['stellar_tuple'])
        cheb_tuple = tuple(read['cheb_tuple'])
        cov_tuple = tuple(read['cov_tuple'])
        region_tuple = tuple(read['region_tuple'])

        #Initialize the Model object
        model = cls(DataSpectrum, Instrument, HDF5Interface, ErrorHDF5Interface, stellar_tuple=stellar_tuple,
                    cheb_tuple=cheb_tuple, cov_tuple=cov_tuple, region_tuple=region_tuple)

        #Update all of the parameters so covariance matrix uploads
        #1) update stellar parameters
        model.update_Model(read['stellar_params'])

        #2) Figure out how many orders, and for each order
        orders_dict = read["orders"]
        #print("orders_dict is", orders_dict)
        orders = [int(i) for i in orders_dict.keys()]
        orders.sort()
        fake_priors = {"sigma0": 5., "mu_width": 2., "sigma_knee" : 150, "frac_global":0.5}
        for i, order in enumerate(orders):
            order_model = model.OrderModels[i]
            order_dict = orders_dict[str(order)]
            #print("order_dict is", order_dict)
            #2.1) update cheb and global cov parametersorder_dict = orders_dict[order]
            order_model.update_Cheb(order_dict['cheb'])
            order_model.update_Cov(order_dict['global_cov'])

            #2.2) instantiate and create all regions, if any exist
            regions_dict = order_dict['regions']
            regions = [int(i) for i in regions_dict.keys()]
            regions.sort()
            if len(regions_dict) > 0:
                #Create regions, otherwise skip
                CovMatrix = order_model.CovarianceMatrix
                for i, region in enumerate(regions):
                    print("creating region ", i, region, regions_dict[str(region)])
                    CovMatrix.create_region(regions_dict[str(region)], fake_priors)

        #Now update the stellar model again so it accounts for the Chebyshevs when downsampling
        model.update_Model(read['stellar_params'])

        return model

    def __init__(self, DataSpectrum, Instrument, HDF5Interface, ErrorHDF5Interface, stellar_tuple, cheb_tuple,
                 cov_tuple, region_tuple, outdir="", max_v=20, ismaster=False, debug=False):
        self.DataSpectrum = DataSpectrum
        self.ismaster = ismaster #Is this the first model instantiated?
        self.stellar_tuple = stellar_tuple
        self.cheb_tuple = cheb_tuple
        self.cov_tuple = cov_tuple
        self.region_tuple = region_tuple
        self.outdir = outdir
        self.debug = debug
        self.orders = self.DataSpectrum.orders
        self.norders = self.DataSpectrum.shape[0]
        #Determine whether `alpha` is in the `stellar_tuple`, then choose trilinear.
        if 'alpha' not in self.stellar_tuple:
            trilinear = True
        else:
            trilinear = False
        #myInterpolator = Interpolator(HDF5Interface, self.DataSpectrum, trilinear=trilinear)
        fluxInterpolator = Interpolator(HDF5Interface, self.DataSpectrum, trilinear=trilinear)
        errorInterpolator = ErrorInterpolator(ErrorHDF5Interface, self.DataSpectrum, trilinear=trilinear)

        self.ModelSpectrum = ModelSpectrum(fluxInterpolator, errorInterpolator, Instrument)
        self.stellar_params = None
        self.stellar_params_last = None
        self.logPrior = 0.0
        self.logPrior_last = 0.0

        self.logger = logging.getLogger(self.__class__.__name__)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        #Now create a a list which contains an OrderModel for each order
        self.OrderModels = [OrderModel(self.ModelSpectrum, self.DataSpectrum, index, max_v=max_v,
                                       npoly=len(self.cheb_tuple), debug=self.debug)
                            for index in range(self.norders)]


    def zip_stellar_p(self, p):
        return dict(zip(self.stellar_tuple, p))

    def zip_Cheb_p(self, p):
        return dict(zip(self.cheb_tuple, p))

    def zip_Cov_p(self, p):
        return dict(zip(self.cov_tuple, p))

    def zip_Region_p(self, p):
        return dict(zip(self.region_tuple, p))

    def update_Model(self, params):
        '''
        Update the model to reflect the stellar parameters
        '''
        self.stellar_params_last = self.stellar_params
        self.stellar_params = params
        self.ModelSpectrum.update_all(params)
        #print("ModelSpectrum.update_all")

        if self.ismaster:
            self.logPrior = self.evaluate_logPrior(params)

        #Since the ModelSpectrum fluxes have been updated, also update the interpolation errors

        #print("Sum of errors is {}".format(np.sum(model_errs)))
        for orderModel in self.OrderModels:
            errs = self.ModelSpectrum.downsampled_errors[:, orderModel.index, :].copy()
            assert errs.flags["C_CONTIGUOUS"], "Not C contiguous"
            orderModel.CovarianceMatrix.update_interp_errs(errs)

    def revert_Model(self):
        '''
        Undo the most recent change to the stellar parameters
        '''
        #Reset stellar_params
        self.stellar_params = self.stellar_params_last
        if self.ismaster:
            self.logPrior = self.logPrior_last
        #Reset downsampled flux
        self.ModelSpectrum.revert_flux()
        #Since the interp_errors have been updated, revert them now
        for orderModel in self.OrderModels:
            orderModel.CovarianceMatrix.revert_interp()

    def get_data(self):
        '''
        Returns a DataSpectrum object.
        '''
        return self.DataSpectrum

    def evaluate(self):
        '''
        Compare the different data and models.
        '''
        self.logger.debug("evaluating model {}".format(self))
        lnps = np.empty((self.norders,))

        for i in range(self.norders):
            #Evaluate using the current CovarianceMatrix
            lnps[i] = self.OrderModels[i].evaluate()

        return np.sum(lnps) + self.logPrior

    def evaluate_logPrior(self, params):
        '''
        Define the prior here
        '''
        logg = params["logg"]
        #if (logg >= 4.8) and (logg <= 5.2):
        return -0.5 * (logg - 5.0)**2/(0.15)**2
        #else:
        #    return -np.inf

    def to_json(self, fname="model.json"):
        '''
        Write all of the available parameters to a JSON file so that we may go back and re-create the model.
        '''

        f = open(self.outdir + fname, 'w')
        json.dump(self, f, cls=ModelEncoder, indent=2, sort_keys=True)
        f.close()

class ModelHA:
    '''
    This is for testing purposes.
    Container class to create and bring together all of the relevant data and models to aid in evaulation.

    :param DataSpectrum: the data to fit
    :type DataSpectrum: :obj:`spectrum.DataSpectrum` object
    :param Instrument: the instrument with which the data was acquired
    :type Instrument: :obj:`grid_tools.Instrument` object
    :param HDF5Interface: the interface to the synthetic stellar library
    :type HDF5Interface: :obj:`grid_tools.HDF5Interface` object
    :param stellar_tuple: describes the order of parameters. If ``alpha`` is missing, :obj:``grid_tools.Interpolator`` is trilinear.
    :type stellar_tuple: tuple

    '''

    @classmethod
    def from_json(cls, filename, DataSpectrum, Instrument, HDF5Interface):
        '''
        Instantiate from a JSON file.
        '''

        #Determine tuples from the JSON output
        f = open(filename, "r")
        read = json.load(f)
        f.close()

        #Read DataSpectrum, Instrument, HDF5Interface, stellar_tuple, cov_tuple, and region_tuple
        stellar_tuple = tuple(read['stellar_tuple'])
        cheb_tuple = tuple(read['cheb_tuple'])
        cov_tuple = tuple(read['cov_tuple'])
        region_tuple = tuple(read['region_tuple'])

        #Initialize the Model object
        model = cls(DataSpectrum, Instrument, HDF5Interface, stellar_tuple=stellar_tuple, cheb_tuple=cheb_tuple,
                    cov_tuple=cov_tuple, region_tuple=region_tuple)

        #Update all of the parameters so covariance matrix uploads
        #1) update stellar parameters
        model.update_Model(read['stellar_params'])

        #2) Figure out how many orders, and for each order
        orders_dict = read["orders"]
        #print("orders_dict is", orders_dict)
        orders = [int(i) for i in orders_dict.keys()]
        orders.sort()
        for i, order in enumerate(orders):
            order_model = model.OrderModels[i]
            order_dict = orders_dict[str(order)]
            #print("order_dict is", order_dict)
            #2.1) update cheb and global cov parametersorder_dict = orders_dict[order]
            order_model.update_Cheb(order_dict['cheb'])
            order_model.update_Cov(order_dict['global_cov'])

            #2.2) instantiate and create all regions, if any exist
            regions_dict = order_dict['regions']
            regions = [int(i) for i in regions_dict.keys()]
            regions.sort()
            if len(regions_dict) > 0:
                #Create regions, otherwise skip
                CovMatrix = order_model.CovarianceMatrix
                for i, region in enumerate(regions):
                    print("creating region ", i, region, regions_dict[str(region)])
                    CovMatrix.create_region(regions_dict[str(region)])

        #Now update the stellar model again so it accounts for the Chebyshevs when downsampling
        model.update_Model(read['stellar_params'])

        return model

    def __init__(self, DataSpectrum, Instrument, HDF5Interface, stellar_tuple, cheb_tuple, cov_tuple, region_tuple, outdir=""):
        self.DataSpectrum = DataSpectrum
        self.stellar_tuple = stellar_tuple
        self.cheb_tuple = cheb_tuple
        self.cov_tuple = cov_tuple
        self.region_tuple = region_tuple
        self.outdir = outdir
        self.orders = self.DataSpectrum.orders
        self.norders = self.DataSpectrum.shape[0]
        #Determine whether `alpha` is in the `stellar_tuple`, then choose trilinear.
        if 'alpha' not in self.stellar_tuple:
            trilinear = True
        else:
            trilinear = False
        myInterpolator = Interpolator(HDF5Interface, self.DataSpectrum, trilinear=trilinear, log=False)
        self.ModelSpectrum = ModelSpectrumHA(myInterpolator, Instrument)
        self.stellar_params = None

        #Now create a a list which contains an OrderModel for each order
        self.OrderModels = [OrderModel(self.ModelSpectrum, self.DataSpectrum, index) for index in range(self.norders)]

    def zip_stellar_p(self, p):
        return dict(zip(self.stellar_tuple, p))

    def zip_Cheb_p(self, p):
        return dict(zip(self.cheb_tuple, p))

    def zip_Cov_p(self, p):
        return dict(zip(self.cov_tuple, p))

    def zip_Region_p(self, p):
        return dict(zip(self.region_tuple, p))

    def update_Model(self, params):
        self.ModelSpectrum.update_all(params)
        self.stellar_params = params

        #Since the ModelSpectrum fluxes have been updated, also update the interpolation errors
        model_errs = self.ModelSpectrum.downsampled_errors
        for orderModel in self.OrderModels:
            errspecs = np.ascontiguousarray(model_errs[:, orderModel.index, :])
            orderModel.CovarianceMatrix.update_interp_errs(errspecs)

    def get_data(self):
        '''
        Returns a DataSpectrum object.
        '''
        return self.DataSpectrum



    def evaluate(self):
        '''
        Compare the different data and models.
        '''
        #Incorporate priors using self.ModelSpectrum.params, self.ChebyshevSpectrum.c0s, cns, self.CovarianceMatrix.params, etc...

        lnps = np.empty((self.norders,))

        for i in range(self.norders):
            #Correct the warp of the model using the ChebyshevSpectrum
            # model_fl = self.OrderModels[i].ChebyshevSpectrum.k * self.ModelSpectrum.downsampled_fls[i]

            #Evaluate using the current CovarianceMatrix
            # lnps[i] = self.OrderModels[i].evaluate(model_fl)
            lnps[i] = self.OrderModels[i].evaluate()

        return np.sum(lnps)


    def to_json(self, fname="model.json"):
        '''
        Write all of the available parameters to a JSON file so that we may go back and re-create the model.
        '''

        f = open(self.outdir + fname, 'w')
        json.dump(self, f, cls=ModelEncoder, indent=2, sort_keys=True)
        f.close()

class OrderModel:
    def __init__(self, ModelSpectrum, DataSpectrum, index, max_v=20, npoly=4, debug=False):
        print("Creating OrderModel {}".format(index))
        self.index = index
        self.DataSpectrum = DataSpectrum
        self.wl = self.DataSpectrum.wls[self.index]
        self.fl = self.DataSpectrum.fls[self.index]
        self.sigma = self.DataSpectrum.sigmas[self.index]
        self.mask = self.DataSpectrum.masks[self.index]
        self.order = self.DataSpectrum.orders[self.index]
        self.debug = debug
        self.logger = logging.getLogger("{} {}".format(self.__class__.__name__, self.order))
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.ModelSpectrum = ModelSpectrum
        self.ChebyshevSpectrum = ChebyshevSpectrum(self.DataSpectrum, self.index, npoly=npoly)
        self.CovarianceMatrix = CovarianceMatrix(self.DataSpectrum, self.index, max_v=max_v, debug=self.debug)
        self.global_cov_params = None
        self.cheb_params = None
        self.cheb_params_last = None
        self.resid_deque = deque(maxlen=500) #Deque that stores the last residual spectra, for averaging
        self.counter = 0

    def get_data(self):
        return (self.wl, self.fl, self.sigma, self.mask)

    def update_Cheb(self, params):
        self.cheb_params_last = self.cheb_params
        self.ChebyshevSpectrum.update(params)
        self.cheb_params = params

    def revert_Cheb(self):
        '''
        Revert the Chebyshev polynomial to a previous state.
        '''
        self.cheb_params = self.cheb_params_last
        self.ChebyshevSpectrum.revert()

    def get_Cheb(self):
        return self.ChebyshevSpectrum.k

    def update_Cov(self, params):
        self.CovarianceMatrix.update_global(params)
        self.global_cov_params = params

    def revert_Cov(self):
        '''
        Since it is too hard to revert the individual parameter changes, instead we just reset the Covariance Matrix
        to it's previous factored state.
        '''
        #Reverts logdet, L, and logPrior
        self.CovarianceMatrix.revert_global()
        #self.CovarianceMatrix.revert()

    def get_Cov(self):
        return self.CovarianceMatrix.cholmod_to_scipy_sparse()

    def get_Cov_interp(self):
        return self.CovarianceMatrix.cholmod_to_scipy_sparse_interp()

    def get_regions_dict(self):
        return self.CovarianceMatrix.get_regions_dict()

    def get_spectrum(self):
        return self.ChebyshevSpectrum.k * self.ModelSpectrum.downsampled_fls[self.index]

    def get_residuals(self):
        model_fl = self.ChebyshevSpectrum.k * self.ModelSpectrum.downsampled_fls[self.index]
        return self.fl - model_fl

    def get_residual_array(self):
        if len(self.resid_deque) > 0:
            return np.array(self.resid_deque)
        else:
            return None

    def clear_resid_deque(self):
        self.resid_deque.clear()

    def evaluate_region_logic(self):
        print("evaluating region logic")
        #calculate the current amplitude of the global_covariance noise
        global_amp = self.CovarianceMatrix.get_amp()
        print("Global amplitude is {}".format(global_amp))
        print("3 times is {}".format(3. * global_amp))

        #array that specifies whether any given pixel is covered by a region.
        covered = self.CovarianceMatrix.get_region_coverage()
        print("There are {} pixels covered by regions.".format(np.sum(covered)))

        residuals = self.get_residuals()
        #median = np.median(residuals)
        #For each residual, calculate the abs_distance from the median
        #from zero
        #abs_distances = np.abs(residuals - median)
        abs_distances = np.abs(residuals)

        #Count how many pixels are above 3 * amplitude
        print("There are {} pixels above 3 * global amp".format(np.sum(abs_distances > (3 * global_amp))))

        #Sort the list in decreasing order of abs_dist
        args = reversed(np.argsort(abs_distances))

        for index in args:
            abs_dist = abs_distances[index]
            if abs_dist < (3 * global_amp):
                #we have reached below the 3 sigma limit, no new regions instantiated
                self.logger.debug("Reached below 3 times limit, no new regions to instantiate.")
                return None

            wl = self.wl[index]
            self.logger.debug("At wl={:.3f}, residual={}".format(wl, abs_dist))
            if covered[index]:
                self.logger.debug("residual already covered")
                continue
            else:
                self.logger.debug("returning wl to instantiate")
                return wl
        else:
            self.logger.warning("Reached the end of the for loop, ie, iterated through all of the residuals.")
            return None

    def evaluate(self):
        '''
        Compare the different data and models.
        '''

        model_fl = self.ChebyshevSpectrum.k * self.ModelSpectrum.downsampled_fls[self.index]

        residuals = self.fl - model_fl
        residuals = residuals[self.mask]

        self.counter += 1
        if self.counter == 100:
            self.resid_deque.append(residuals)
            self.counter = 0

        lnp = self.CovarianceMatrix.evaluate(residuals)
        return lnp

class Sampler(GibbsSampler):
    '''
    Subclasses the GibbsSampler in emcee

    :param model: the :obj:`Model`
    :param starting_param_dict: the dictionary of starting parameters
    :param cov: the MH proposal


        Optional, specify by kwargs
        order_index
        args = []

    '''

    def __init__(self, **kwargs):
        self.dim = len(self.param_tuple)
        p0 = np.empty((self.dim,))
        starting_param_dict = kwargs.get("starting_param_dict")
        for i,param in enumerate(self.param_tuple):
            p0[i] = starting_param_dict[param]

        kwargs.update({"p0":p0, "dim":self.dim})
        self.model_list = kwargs.get("model_list")
        self.nmodels = len(self.model_list)
        if "model_index" in kwargs:
            self.model_index = kwargs.get("model_index")
            self.model = self.model_list[self.model_index]

        super(Sampler, self).__init__(**kwargs)

        #Each subclass will have to overwrite how it parses the param_dict into the correct order
        #and sets the param_tuple

        #SUBCLASS here and define self.param_tuple
        #SUBCLASS here and define self.lnprob
        #SUBCLASS here and do self.revertfn
        #then do super().__init__() to call the following code

        self.outdir = kwargs.get("outdir", "")

    def lnprob(self):
        raise NotImplementedError("To be implemented by a subclass!")

    def revertfn(self):
        raise NotImplementedError("To be implemented by a subclass!")

    def write(self):
        '''
        Write all of the relevant sample output to an HDF file.

        Write the lnprobability to an HDF file.

        flatchain
        acceptance fraction
        tuple parameters as an attribute in the header from self.param_tuple

        The actual HDF5 file is structured as follows

        /
        stellar parameters.flatchain
        00/
        ...
        22/
        23/
            global_cov.flatchain
            regions/
                region1.flatchain

        Everything can be saved in the dataset self.fname

        '''

        filename = self.outdir + "flatchains.hdf5"
        self.logger.debug("Opening {} for writing HDF5 flatchains".format(filename))
        hdf5 = h5py.File(filename, "a") #creates if doesn't exist, otherwise read/write
        samples = self.flatchain

        self.logger.debug("Creating dataset with fname:{}".format(self.fname))
        dset = hdf5.create_dataset(self.fname, samples.shape, compression='gzip', compression_opts=9)
        self.logger.debug("Storing samples and header attributes.")
        dset[:] = samples
        dset.attrs["parameters"] = "{}".format(self.param_tuple)
        dset.attrs["acceptance"] = "{}".format(self.acceptance_fraction)
        dset.attrs["acor"] = "{}".format(self.acor)
        dset.attrs["commit"] = "{}".format(C.get_git_commit())
        hdf5.close()

        #lnprobability is the lnprob at each sample
        filename = self.outdir + "lnprobs.hdf5"
        self.logger.debug("Opening {} for writing HDF5 lnprobs".format(filename))
        hdf5 = h5py.File(filename, "a") #creates if doesn't exist, otherwise read/write
        lnprobs = self.lnprobability

        dset = hdf5.create_dataset(self.fname, samples.shape[:1], compression='gzip', compression_opts=9)
        dset[:] = lnprobs
        dset.attrs["commit"] = "{}".format(C.get_git_commit())
        hdf5.close()

    def plot(self):
        '''
        Generate the relevant plots once the sampling is done.
        '''
        samples = self.flatchain

        plot_walkers(self.outdir + self.fname + "_chain_pos.png", samples, labels=self.param_tuple)

        import triangle
        figure = triangle.corner(samples, labels=self.param_tuple, quantiles=[0.16, 0.5, 0.84],
                                 show_titles=True, title_args={"fontsize": 12})
        figure.savefig(self.outdir + self.fname + "_triangle.png")

        plt.close(figure)

class StellarSampler(Sampler):
    """
    Subclasses the Sampler specifically for stellar parameters

    :param model:
        the :obj:`Model`

    :param starting_param_dict:
        the dictionary of starting parameters

    :param cov:
        the MH proposal

    :param fix_logg:
        fix logg? If so, to what value?

    :param order_index: order index

    :param args: []

    """
    def __init__(self, **kwargs):
        self.fix_logg = kwargs.get("fix_logg", False)

        starting_param_dict = kwargs.get("starting_param_dict")
        self.param_tuple = C.dictkeys_to_tuple(starting_param_dict)
        print("param_tuple is {}".format(self.param_tuple))

        kwargs.update({"revertfn":self.revertfn, "lnprobfn":self.lnprob})
        super(StellarSampler, self).__init__(**kwargs)
        self.fname = "stellar"

        #From now on, self.model will always be a list of models (or just a list with one model)

    def reset(self):
        #Clear accumulated residuals in each order sampler
        for model in self.model_list:
            for order_model in model.OrderModels:
                order_model.clear_resid_deque()
        super(StellarSampler, self).reset()

    def revertfn(self):
        '''
        Revert the model to the previous state of parameters, in the case of a rejected MH proposal.
        '''
        self.logger.debug("reverting stellar parameters")
        for model in self.model_list:
            model.revert_Model()

    def lnprob(self, p):
        # We want to send the same stellar parameters to each model, but also the different vz and logOmega parameters
        # to the separate models.
        self.logger.debug("lnprob p is {}".format(p))

        #Extract only the temp, logg, Z, vsini parameters
        if not self.fix_logg:
            params = self.model_list[0].zip_stellar_p(p[:4])
            others = p[4:]
        else:
            #Coming in as temp, Z, vsini, vz, logOmega...

            params = self.model_list[0].zip_stellar_p(p[:3])
            others = p[3:]
            params.update({"logg": self.fix_logg})

        self.logger.debug("params in lnprob are {}".format(params))

        #others should now be either [vz, logOmega] or [vz0, logOmega0, vz1, logOmega1, ...] etc. Always div by 2.
        #split p up into [vz, logOmega], [vz, logOmega] pairs that update the other parameters.
        #mparams is now a list of parameter dictionaries
        mparams = deque()
        for vz, logOmega in grouper(others, 2):
            p = params.copy()
            p.update({"vz":vz, "logOmega":logOmega})
            mparams.append(p)

        self.logger.debug("updated lnprob params: {}".format([params for params in mparams]))
        try:

            lnps = np.empty((self.nmodels,))
            for i, (model, par) in enumerate(zip(self.model_list, mparams)):
                self.logger.debug("Updating model {}:{} with {}".format(i, model, par))
                model.update_Model(par) #This also updates downsampled_fls
                #For order in myModel, do evaluate, and sum the results.
                lnp = model.evaluate()
                self.logger.debug("model {}:{} lnp {}".format(i, model, lnp))
                lnps[i] = lnp
            self.logger.debug("lnps : {}".format(lnps))
            s = np.sum(lnps)
            self.logger.debug("sum lnps {}".format(s))
            return s
        except C.ModelError:
            self.logger.debug("lnprob returning -np.inf")
            return -np.inf

class ChebSampler(Sampler):
    def __init__(self, **kwargs):

        starting_param_dict = kwargs.get("starting_param_dict")
        nparams = len(starting_param_dict)
        self.param_tuple = ("logc0",) + tuple(["c{}".format(i) for i in range(1, nparams)])

        kwargs.update({"revertfn":self.revertfn, "lnprobfn":self.lnprob})
        super(ChebSampler, self).__init__(**kwargs)

        self.order_index = kwargs.get("order_index")
        self.fname = "{}/{}/{}".format(self.model_index, self.model.orders[self.order_index], "cheb")
        self.order_model = self.model.OrderModels[self.order_index]

    def revertfn(self):
        self.logger.debug("reverting model")
        self.order_model.revert_Cheb()

    def lnprob(self, p):
        params = self.model.zip_Cheb_p(p)
        self.logger.debug("Updating cheb params {}".format(params))
        #self.logger.debug("Updating model {}:{}, order {} with params {}".format(self.model_index,
        #                                                        self.model, self.order_index, params))
        self.order_model.update_Cheb(params)

        lnps = np.empty((self.nmodels,))
        for i, model in enumerate(self.model_list):
            lnps[i] = model.evaluate()
        s = np.sum(lnps)
        #self.logger.debug("lnp {}".format(s))
        return s

class CovGlobalSampler(Sampler):

    def __init__(self, **kwargs):
        starting_param_dict = kwargs.get("starting_param_dict")
        self.param_tuple = C.dictkeys_to_cov_global_tuple(starting_param_dict)

        kwargs.update({"revertfn":self.revertfn, "lnprobfn":self.lnprob})
        super(CovGlobalSampler, self).__init__(**kwargs)

        self.order_index = kwargs.get("order_index")
        self.fname = "{}/{}/{}".format(self.model_index, self.model.orders[self.order_index], "cov")
        self.order_model = self.model.OrderModels[self.order_index]

    def revertfn(self):
        self.logger.debug("reverting model")
        self.order_model.revert_Cov()

    def lnprob(self, p):
        params = self.model.zip_Cov_p(p)
        self.logger.debug("Updating model {}:{}, order {} with params {}".format(self.model_index,
                                self.model, self.order_index, params))
        try:
            self.order_model.update_Cov(params)
            lnps = np.empty((self.nmodels,))
            for i, model in enumerate(self.model_list):
                lnps[i] = model.evaluate()
            s = np.sum(lnps)
            self.logger.debug("lnp {}".format(s))
            return s
        except C.ModelError:
            return -np.inf

class MegaSampler(GibbsController):
    def write(self):
        for sampler in self.samplers:
            sampler.write()

    def plot(self):
        for sampler in self.samplers:
            sampler.plot()

    def instantiate_regions(self, sigma=3):
        for sampler in self.samplers:
        #if it is a Regions Sampler
            if type(sampler) is RegionsSampler:
                sampler.instantiate_regions(sigma)

    def trim_regions(self):
        for sampler in self.samplers:
            #if it is a Regions Sampler
            if type(sampler) is RegionsSampler:
                sampler.trim_regions()

class CovRegionSampler(Sampler):
    def __init__(self, **kwargs):
        starting_param_dict = kwargs.get("starting_param_dict")
        self.priors = kwargs.get("priors")
        self.param_tuple = ("loga", "mu", "sigma")

        kwargs.update({"revertfn":self.revertfn, "lnprobfn":self.lnprob})
        super(CovRegionSampler, self).__init__(**kwargs)

        self.order_index = kwargs.get("order_index")
        self.region_index = kwargs.get("region_index")
        self.fname = "{}/{}/{}{:0>2}".format(self.model_index, self.model.orders[self.order_index], "cov_region",
                                             self.region_index)
        self.order_model = self.model.OrderModels[self.order_index]
        self.CovMatrix = self.order_model.CovarianceMatrix
        self.CovMatrix.create_region(starting_param_dict, self.priors)
        self.a = 10**starting_param_dict['loga']
        self.a_last = self.a
        self.logger.info("Created new Region sampler with region_index {}".format(self.region_index))

    def __del__(self):
        '''
        Delete this sampler from the list and the RegionCovarianceMatrix which it references.
        '''
        self.CovMatrix.delete_region(self.region_index)

    def revertfn(self):
        self.logger.debug("reverting model")
        self.CovMatrix.revert_region(self.region_index)
        self.a = self.a_last

    def lnprob(self, p):
        params = self.model.zip_Region_p(p)

        #We need to do things like this and not immediately exit because we shouldn't do two reverts in a row without
        #  at least trying to fill the matrix.
        a_global = self.order_model.CovarianceMatrix.get_amp()
        self.a_last = self.a
        self.a = 10**params["loga"]

        try:
            self.CovMatrix.update_region(self.region_index, params)
            lnps = np.empty((self.nmodels,))
            for i, model in enumerate(self.model_list):
                lnps[i] = model.evaluate()
            s = np.sum(lnps)
            self.logger.debug("lnp {}".format(s))
            #Institute the hard prior that the amplitude can't be less than the Global Covariance
            if self.a < (self.priors['frac_global'] * a_global):
                return -np.inf
            else:
                return s
        except (C.ModelError, C.RegionError):
            return -np.inf

class RegionsSampler(GibbsSubController):
    '''
    Designed to accumulate CovRegionSampler's in a list.
    '''
    def __init__(self, **kwargs):
        self.max_regions = kwargs.get("max_regions", 0)
        self.default_param_dict = kwargs.get("default_param_dict", {"loga":-14.2, "sigma":10.})
        self.priors = kwargs.get("priors")
        self.order_index = kwargs.get("order_index")
        self.model_list = kwargs.get("model_list")
        self.nmodels = len(self.model_list)
        if "model_index" in kwargs:
            self.model_index = kwargs.get("model_index")
            self.model = self.model_list[self.model_index]
        self.order_model = self.model.OrderModels[self.order_index]
        self.MH_cov = kwargs.get("cov")
        self.outdir = kwargs.get("outdir")
        self.region_index = 0

        kwargs.update({"samplers":[]}) #start empty and update later
        super(RegionsSampler, self).__init__(**kwargs)

    def evaluate_region_logic(self):
        return self.order_model.evaluate_region_logic()

    def create_region_sampler(self, mu):
        #mu is the location just returned by evaluate_region_logic().
        #Create a new region in the model at this location, create a new CovRegionSampler to sample it.

        #copy and update default_param_dict to include the new value of mu
        starting_param_dict = self.default_param_dict.copy()
        starting_param_dict.update({"mu":mu})

        newSampler = CovRegionSampler(model_list=self.model_list, model_index=self.model_index, cov=self.MH_cov,
                starting_param_dict=starting_param_dict, priors=self.priors, order_index=self.order_index,
                region_index=self.region_index, outdir=self.outdir, debug=self.debug)
        self.samplers.append(newSampler)
        print("Now there are {} region samplers.".format(len(self.samplers)))
        self.region_index += 1 #increment for next region to be instantiated.

    def trim_regions(self):
        print("Trimming Regions with amplitudes below 2.5 sigma.")
        #Go through and find the amplitude of each region. If it's below 2*global amplitude, delete the sampler from
        # the list, subsequently deleting the region matrix.
        a_global = self.order_model.CovarianceMatrix.get_amp()
        print("a_global is {}".format(a_global))
        print("2.5 cutoff is {}".format(2.5 * a_global))
        print("3 times is {}".format(3. * a_global))
        for i, sampler in enumerate(self.samplers):
            print("sampler.a is {}".format(sampler.a))
            if sampler.a < 2.5 * a_global:
                print("Deleting region sampler from list.")
                del self.samplers[i] #Also triggers sampler.__del__()
        print("Now there are only {} region samplers left.".format(len(self.samplers)))

    def instantiate_regions(self, sigma=3.):
        #array that specifies if a pixel is already covered. On the first call, it should be all False
        covered = self.order_model.CovarianceMatrix.get_region_coverage()

        #average all of the spectra in the deque together
        residual_array = self.order_model.get_residual_array()
        if residual_array is None:
            residuals = self.order_model.get_residuals()
        else:
            residuals = np.average(residual_array, axis=0)

        #run the sigma_clip algorithm until converged, and we've identified the outliers
        filtered_data = sigma_clip(residuals, sig=sigma, iters=None)
        mask = filtered_data.mask
        wl = self.order_model.wl

        #Sort in decreasing strength of residual
        for w,resid in sorted(zip(wl[mask], np.abs(residuals[mask])), key=itemgetter(1), reverse=True):
            if w in wl[covered]:
                continue
            else:
                #check to make sure region is not *right* at the edge of the spectrum
                if w <= np.min(wl) or w >= np.max(wl):
                    continue
                else:
                    #instantiate region and update coverage
                    self.create_region_sampler(w)
                    covered = self.order_model.CovarianceMatrix.get_region_coverage()

    # def run_mcmc(self, *args, **kwargs):
    #
    #     if (len(self.samplers) < self.max_regions):
    #         mu = self.evaluate_region_logic()
    #         if mu is not None:
    #             self.create_region_sampler(mu)
    #     result = super(RegionsSampler, self).run_mcmc(*args, **kwargs)
    #     return result

    def write(self):
        for sampler in self.samplers:
            sampler.write()

    def plot(self):
        for sampler in self.samplers:
            sampler.plot()

class OldMegaSampler:
    '''
    One Sampler to rule them all

    :param samplers: the various Sampler objects to sample
    :type samplers: list or tuple of :obj:`model.Sampler`s
    :param cadence: of sub-iterations per iteration
    :type cadence: list or tuple


    '''
    def __init__(self, model, samplers, burnInCadence, cadence):
        self.model = model
        self.samplers = samplers
        self.nsamplers = len(self.samplers)
        self.burnInCadence = burnInCadence
        self.cadence = cadence
        self.json_dump = 10 #Number of times to dump to output

    def burn_in(self, iterations):
        for i in range(iterations):
            print("\n\nMegasampler on burn in iteration {} of {}".format(i, iterations))
            for j in range(self.nsamplers):
                #self.samplers[j].run(self.burnInCadence[j])
                self.samplers[j].burn_in(self.burnInCadence[j])

    def reset(self):
        for j in range(self.nsamplers):
            self.samplers[j].reset()

    def run(self, iterations):
        #Choose 10 random numbers for which to dump output
        dump_indexes = np.random.randint(iterations, size=(self.json_dump,))
        print("Will dump output on iterations", dump_indexes)

        for i in range(iterations):
            print("\n\nMegasampler on iteration {} of {}".format(i, iterations))
            for j in range(self.nsamplers):
                self.samplers[j].run(self.cadence[j])
            if i in dump_indexes:
                self.model.to_json("model{:0>2}.json".format(i))


    def write(self):
        for j in range(self.nsamplers):
            self.samplers[j].write()

    def plot(self):
        for j in range(self.nsamplers):
            self.samplers[j].plot()

    def acceptance_fraction(self):
        for j in range(self.nsamplers):
            print(self.samplers[j].acceptance_fraction())

    def acor(self):
        for j in range(self.nsamplers):
            print(self.samplers[j].acor())

def main():
    print("Starting main of model")

    pass

if __name__ == "__main__":
    main()
