import numpy as np
import sys
import emcee
from . import constants as C
from .grid_tools import Interpolator
from .spectrum import ModelSpectrum, ChebyshevSpectrum, ModelSpectrumHA
from .covariance import CovarianceMatrix #from StellarSpectra.spectrum import CovarianceMatrix #pure Python
import time
import json
import h5py
import os
import matplotlib.pyplot as plt

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
        myInterpolator = Interpolator(HDF5Interface, self.DataSpectrum, trilinear=trilinear)
        self.ModelSpectrum = ModelSpectrum(myInterpolator, Instrument)
        self.stellar_params = None

        #Now create a a list which contains an OrderModel for each order
        self.OrderModels = [OrderModel(self.ModelSpectrum, self.DataSpectrum, index, npoly=len(self.cheb_tuple))
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
        self.ModelSpectrum.update_all(params)
        self.stellar_params = params

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
    def __init__(self, ModelSpectrum, DataSpectrum, index, npoly=4):
        print("Creating OrderModel {}".format(index))
        self.index = index
        self.DataSpectrum = DataSpectrum
        self.wl = self.DataSpectrum.wls[self.index]
        self.fl = self.DataSpectrum.fls[self.index]
        self.sigma = self.DataSpectrum.sigmas[self.index]
        self.mask = self.DataSpectrum.masks[self.index]
        self.order = self.DataSpectrum.orders[self.index]
        self.ModelSpectrum = ModelSpectrum
        self.ChebyshevSpectrum = ChebyshevSpectrum(self.DataSpectrum, self.index, npoly=npoly)
        self.CovarianceMatrix = CovarianceMatrix(self.DataSpectrum, self.index)
        self.global_cov_params = None
        self.cheb_params = None

        #We can expose the RegionMatrices from the self.CovarianceMatrix, or keep track as they are added
        self.region_list = []

    def get_data(self):
        return (self.wl, self.fl, self.sigma, self.mask)

    def update_Cheb(self, params):
        self.ChebyshevSpectrum.update(params)
        self.cheb_params = params

    def get_Cheb(self):
        return self.ChebyshevSpectrum.k

    def update_Cov(self, params):
        self.CovarianceMatrix.update_global(params)
        self.global_cov_params = params

    def get_Cov(self):
        return self.CovarianceMatrix.cholmod_to_scipy_sparse()

    def get_regions_dict(self):
        return self.CovarianceMatrix.get_regions_dict()

    def get_spectrum(self):
        return self.ChebyshevSpectrum.k * self.ModelSpectrum.downsampled_fls[self.index]

    def get_residuals(self):
        model_fl = self.ChebyshevSpectrum.k * self.ModelSpectrum.downsampled_fls[self.index]
        return self.fl - model_fl

    def evaluate_region_logic(self):
        #calculate the current amplitude of the global_covariance noise
        global_amp = self.CovarianceMatrix.get_amp()

        #array that specifies whether any given pixel is covered by a region.
        covered = self.CovarianceMatrix.get_region_coverage()

        residuals = self.get_residuals()
        median = np.median(residuals)
        #For each residual, calculate the abs_distance from the median
        abs_distances = np.abs(residuals - median)

        #Sort the list in decreasing order of abs_dist
        args = np.argsort(abs_distances)[::-1]

        for index in args:
            abs_dist = abs_distances[index]
            if abs_dist < 3 * global_amp:
            #we have reached below the 3 sigma limit, no new regions instantiated
                return None

            wl = self.wl[index]
            print("At wl={:.3f}, residual={}".format(wl, abs_dist))
            if covered[index]:
                continue
            else:
                return wl

        return None

    def evaluate(self):
        '''
        Compare the different data and models.
        '''
        #Incorporate priors using self.ModelSpectrum.params, self.ChebyshevSpectrum.c0s, cns, self.CovarianceMatrix.params, etc...

        model_fl = self.ChebyshevSpectrum.k * self.ModelSpectrum.downsampled_fls[self.index]

        model_fl = model_fl[self.mask]

        #CovarianceMatrix will do the lnprob math without priors
        lnp = self.CovarianceMatrix.evaluate(model_fl)
        return lnp

class Sampler:
    '''
    Helper class designed to be overwritten for StellarSampler, ChebSampler, CovSampler.C

    '''
    def __init__(self, model, MH_cov, starting_param_dict, outdir="", fname="", order_index=None):
        #Each subclass will have to overwrite how it parses the param_dict into the correct order
        #and sets the param_tuple

        #SUBCLASS OVERWRITE HERE to create self.param_tuple

        #SUBCLASS here to write self.lnprob

        self.model = model
        self.ndim = len(self.param_tuple)

        self.p0 = np.empty((self.ndim))
        for i,param in enumerate(self.param_tuple):
            self.p0[i] = starting_param_dict[param]

        if order_index is None:
            self.sampler = emcee.MHSampler(MH_cov, self.ndim, self.lnprob)
        else:
            self.sampler = emcee.MHSampler(MH_cov, self.ndim, self.lnprob, args=(order_index,))

        self.pos_trio = None
        self.outdir = outdir
        self.fname = fname

    def reset(self):
        self.sampler.reset()
        print("Reset {}".format(self.param_tuple))

    def run(self, iterations):
        if iterations == 0:
            return
        print("Sampling {} for {} iterations: ".format(self.param_tuple, iterations))
        t = time.time()

        if self.pos_trio is None:
            self.pos_trio = self.sampler.run_mcmc(self.p0, iterations)

        else:
            print("running for {} iterations".format(iterations))
            pos, prob, state = self.pos_trio
            self.pos_trio = self.sampler.run_mcmc(pos, iterations, rstate0=state)

        print("completed in {:.2f} seconds".format(time.time() - t))

    def burn_in(self, iterations):
        '''
        For consistencies's sake w/ MegaSampler
        '''
        self.run(iterations)


    def write(self):
        '''
        Write all of the relevant output to an HDF file.

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
        hdf5 = h5py.File(filename, "a") #creates if doesn't exist, otherwise read/write
        samples = self.sampler.flatchain

        dset = hdf5.create_dataset(self.fname, samples.shape, compression='gzip', compression_opts=9)
        dset[:] = samples
        dset.attrs["parameters"] = "{}".format(self.param_tuple)
        dset.attrs["acceptance"] = "{}".format(self.sampler.acceptance_fraction)

        hdf5.close()

    def plot(self):
        '''
        Generate the relevant plots once the sampling is done.
        '''

        import triangle

        samples = self.sampler.flatchain
        figure = triangle.corner(samples, labels=self.param_tuple, quantiles=[0.16, 0.5, 0.84],
                                 show_titles=True, title_args={"fontsize": 12})
        figure.savefig(self.outdir + self.fname + "_triangle.png")

        plot_walkers(self.outdir + self.fname + "_chain_pos.png", samples, labels=self.param_tuple)
        plt.close(figure)



    def acceptance_fraction(self):
        return self.sampler.acceptance_fraction

class StellarSampler(Sampler):
    '''
    Subclass of Sampler for evaluating stellar parameters.
    '''
    def __init__(self, model, MH_cov, starting_param_dict, fix_logg=None, outdir="", fname="stellar"):
        #Parse param_dict to determine which parameters are present as a subset of stellar parameters, then set self.param_tuple

        self.param_tuple = C.dictkeys_to_tuple(starting_param_dict)
        self.fix_logg = fix_logg if fix_logg is not None else False

        super().__init__(model, MH_cov, starting_param_dict, outdir=outdir, fname=fname)


    def lnprob(self, p):
        params = self.model.zip_stellar_p(p)
        #If we have decided to fix logg, sneak update it here.
        if self.fix_logg:
            params.update({"logg": self.fix_logg})
        print("{}".format(params))
        try:
            self.model.update_Model(params) #This also updates downsampled_fls
            #For order in myModel, do evaluate, and sum the results.
            return self.model.evaluate()
        except C.ModelError:
            #print("Stellar returning -np.inf")
            return -np.inf

class ChebSampler(Sampler):
    '''
    Subclass of Sampler for evaluating Chebyshev parameters.
    '''
    def __init__(self, model, MH_cov, starting_param_dict, order_index=None, outdir="", fname="cheb"):
        #Overwrite the Sampler init method to take ranges for c0, c1, c2, c3 etc... that are the same for each order.
        #From a simple param dict, create a more complex param_dict

        #Then set param_tuple
        nparams = len(starting_param_dict)
        self.param_tuple = ("logc0",) + tuple(["c{}".format(i) for i in range(1, nparams)])
        self.order_index = order_index

        # outdir = "{}{}/".format(outdir, model.orders[self.order_index])
        fname = "{}/{}".format(model.orders[self.order_index], fname)

        super().__init__(model, MH_cov, starting_param_dict, outdir=outdir, fname=fname)

        self.order_model = self.model.OrderModels[self.order_index]

    def lnprob(self, p):
        params = self.model.zip_Cheb_p(p)
        print("params are", params)
        self.order_model.update_Cheb(params)
        return self.order_model.evaluate()

class CovGlobalSampler(Sampler):
    '''
    Subclass of Sampler for evaluating GlobalCovarianceMatrix parameters.
    '''
    def __init__(self, model, MH_cov, starting_param_dict, order_index=None, outdir="", fname="cov"):
        #Parse param_dict to determine which parameters are present as a subset of parameters, then set self.param_tuple
        self.param_tuple = C.dictkeys_to_cov_global_tuple(starting_param_dict)

        self.order_index = order_index
        # outdir = "{}{}/".format(outdir, model.orders[self.order_index])
        fname = "{}/{}".format(model.orders[self.order_index], fname)
        super().__init__(model, MH_cov, starting_param_dict, outdir=outdir, fname=fname)

        self.order_model = self.model.OrderModels[self.order_index]

    def lnprob(self, p):
        params = self.model.zip_Cov_p(p)
        # if params["l"] > 1:
        #     return -np.inf
        try:
            self.order_model.update_Cov(params)
            return self.order_model.evaluate()
        except C.ModelError:
            return -np.inf

class CovRegionSampler(Sampler):
    '''
    Subclass of Sampler for evaluating RegionCovarianceMatrix parameters.
    '''
    def __init__(self, model, MH_cov, starting_param_dict, order_index=None, region_index=None, outdir="", fname="cov_region"):
        #Parse param_dict to determine which parameters are present as a subset of parameters, then set self.param_tuple
        self.param_tuple = ("loga", "mu", "sigma")

        self.order_index = order_index
        self.region_index = region_index
        fname += "{:0>2}".format(self.region_index)
        super().__init__(model, MH_cov, starting_param_dict, outdir=outdir, fname=fname)

        self.order_model = self.model.OrderModels[self.order_index]
        self.CovMatrix = self.order_model.CovarianceMatrix
        self.CovMatrix.create_region(starting_param_dict)
        print("Created new Region sampler with region_index {}".format(self.region_index))


    def lnprob(self, p):
        params = self.model.zip_Region_p(p)
        try:
             self.CovMatrix.update_region(self.region_index, params)
             return self.order_model.evaluate()
        except (C.ModelError, C.RegionError):
             return -np.inf

class RegionsSampler:
    '''
    The point of this sampler is to do all the region sampling for a specific order.

    It must also be able to create new samplers.

    There will be a list of RegionSamplers, one for each region. This keys into the specific region properly.

    There will also be some logic that at the beginning of each iteration of the OrderSampler decides
    how to manage the RegionSamplerList.


    '''
    #TODO: subclass RegionsSampler from MegaSampler

    def __init__(self, model, MH_cov, default_param_dict={"loga":-14.2, "sigma":10.}, max_regions = 3,
                 order_index=None,  outdir="", fname="cov_region"):

        self.model = model
        self.MH_cov = MH_cov
        self.default_param_dict = default_param_dict #this is everything but mu
        self.order_index = order_index
        self.order_model = self.model.OrderModels[self.order_index]

        self.cadence = 4 #samples to devote to each region before moving on to the next
        self.logic_counter = 0
        self.logic_overflow = 5 #how many times must RegionsSampler come up in the rotation before we evaluate some logic
        self.max_regions = max_regions
        #to decide if more or fewer RegionSampler's are needed?

        self.samplers = [] #we will add to this list as we instantiate more RegionSampler's
        # outdir = "{}{}/".format(outdir, model.orders[self.order_index])
        self.outdir = outdir
        fname = "{}/{}".format(model.orders[self.order_index], fname)
        self.fname = fname

    def evaluate_region_logic(self):
        return self.order_model.evaluate_region_logic()

    def create_region_sampler(self, mu):
        #mu is the location just returned by evaluate_region_logic(). Therefore we wish to instantiate a new RegionSampler
        #Create a new region in the model at this location, create a new sampler to sample it.

        #copy and update default_param_dict to include the new value of mu
        starting_param_dict = self.default_param_dict.copy()
        starting_param_dict.update({"mu":mu})
        region_index = len(self.samplers) #region_index is one more than the current amount of regions

        newSampler = CovRegionSampler(self.model, self.MH_cov, starting_param_dict, order_index=self.order_index,
                                      region_index=region_index, outdir=self.outdir, fname=self.fname)
        self.samplers.append(newSampler)

    def reset(self):
        for sampler in self.samplers:
            sampler.reset()
            print("Reset RegionsSampler for order {}".format(self.order_index))


    def burn_in(self, iterations):
        '''
        For consistencies's sake w/ MegaSampler
        '''
        self.run(iterations)


    def run(self, iterations):
        if iterations == 0:
            return

        self.logic_counter += 1
        if (self.logic_counter >= self.logic_overflow) and (len(self.samplers) < self.max_regions):
            mu = self.evaluate_region_logic()
            if mu is not None:
                self.create_region_sampler(mu)

        for i in range(iterations):
            for sampler in self.samplers:
                print("Sampling region {} for {} iterations: ".format(sampler.region_index, iterations), end="")
                t = time.time()
                sampler.run(self.cadence)
                print("completed in {:.2f} seconds".format(time.time() - t))

    def write(self):
        for sampler in self.samplers:
            sampler.write()

    def plot(self):
        for sampler in self.samplers:
            sampler.plot()

    def acceptance_fraction(self):
        for j in range(len(self.samplers)):
            print(self.samplers[j].acceptance_fraction())

class MegaSampler:
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


def main():
    print("Starting main of model")

    pass


if __name__ == "__main__":
    main()
