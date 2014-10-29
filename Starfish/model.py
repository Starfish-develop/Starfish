import numpy as np
from emcee import GibbsSampler, ParallelSampler
from . import constants as C
from .grid_tools import Interpolator
from .spectrum import ModelSpectrum, ModelSpectrumHA
import json
import h5py
import logging
import matplotlib.pyplot as plt
from itertools import zip_longest

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def plot_walkers(filename, samples, labels=None):
    ndim = len(samples[0, :])

    figsize = (12, ndim * 1.8)

    fig, ax = plt.subplots(nrows=ndim, sharex=True, figsize=figsize)
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

    def __init__(self, DataSpectrum, Instrument, Emulator, stellar_tuple,
                 cheb_tuple, cov_tuple, region_tuple, outdir="", max_v=20, ismaster=False, debug=False):
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

        Emulator.determine_chunk_log(self.DataSpectrum.wls.flatten()) #Possibly truncate the grid

        self.ModelSpectrum = ModelSpectrum(Emulator, self.DataSpectrum, Instrument)
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
        # for orderModel in self.OrderModels:
        #     errs = self.ModelSpectrum.downsampled_errors[:, orderModel.index, :].copy()
        #     assert errs.flags["C_CONTIGUOUS"], "Not C contiguous"
        #     orderModel.CovarianceMatrix.update_interp_errs(errs)

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
        # for orderModel in self.OrderModels:
        #     orderModel.CovarianceMatrix.revert_interp()

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
        logg_prior = -0.5 * (logg - 5.0)**2/(0.05)**2

        return logg_prior

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

class Sampler(GibbsSampler):
    '''
    Subclasses the GibbsSampler in emcee

    :param cov:
    :param starting_param_dict: the dictionary of starting parameters
    :param cov: the MH proposal
    :param revertfn:
    :param acceptfn:
    :param debug:

    '''

    def __init__(self, **kwargs):
        self.dim = len(self.param_tuple)
        #p0 = np.empty((self.dim,))
        #starting_param_dict = kwargs.get("starting_param_dict")
        #for i,param in enumerate(self.param_tuple):
        #    p0[i] = starting_param_dict[param]

        kwargs.update({"dim":self.dim})
        #self.spectra_list = kwargs.get("spectra_list", [0])

        super(Sampler, self).__init__(**kwargs)

        #Each subclass will have to overwrite how it parses the param_dict into the correct order
        #and sets the param_tuple

        #SUBCLASS here and define self.param_tuple
        #SUBCLASS here and define self.lnprob
        #SUBCLASS here and do self.revertfn
        #then do super().__init__() to call the following code

        self.outdir = kwargs.get("outdir", "")

    def startdict_to_tuple(self, startdict):
        raise NotImplementedError("To be implemented by a subclass!")

    def zip_p(self, p):
        return dict(zip(self.param_tuple, p))

    def lnprob(self):
        raise NotImplementedError("To be implemented by a subclass!")

    def revertfn(self):
        raise NotImplementedError("To be implemented by a subclass!")

    def acceptfn(self):
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

    def plot(self, triangle_plot=False):
        '''
        Generate the relevant plots once the sampling is done.
        '''
        samples = self.flatchain

        plot_walkers(self.outdir + self.fname + "_chain_pos.png", samples, labels=self.param_tuple)

        if triangle_plot:
            import triangle
            figure = triangle.corner(samples, labels=self.param_tuple, quantiles=[0.16, 0.5, 0.84],
                                     show_titles=True, title_args={"fontsize": 12})
            figure.savefig(self.outdir + self.fname + "_triangle.png")

            plt.close(figure)

class PSampler(ParallelSampler):
    '''
    Subclasses the GibbsSampler in emcee

    :param cov:
    :param starting_param_dict: the dictionary of starting parameters
    :param cov: the MH proposal
    :param revertfn:
    :param acceptfn:
    :param debug:

    '''

    def __init__(self, **kwargs):
        self.dim = len(self.param_tuple)
        #p0 = np.empty((self.dim,))
        #starting_param_dict = kwargs.get("starting_param_dict")
        #for i,param in enumerate(self.param_tuple):
        #    p0[i] = starting_param_dict[param]

        kwargs.update({"dim":self.dim})
        #self.spectra_list = kwargs.get("spectra_list", [0])

        super(PSampler, self).__init__(**kwargs)

        #Each subclass will have to overwrite how it parses the param_dict into the correct order
        #and sets the param_tuple

        #SUBCLASS here and define self.param_tuple
        #SUBCLASS here and define self.lnprob
        #SUBCLASS here and do self.revertfn
        #then do super().__init__() to call the following code

        self.outdir = kwargs.get("outdir", "")
    
    def startdict_to_tuple(self, startdict):
        raise NotImplementedError("To be implemented by a subclass!")

    def zip_p(self, p):
        return dict(zip(self.param_tuple, p))

    def lnprob(self):
        raise NotImplementedError("To be implemented by a subclass!")

    def revertfn(self):
        raise NotImplementedError("To be implemented by a subclass!")

    def acceptfn(self):
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

    def plot(self, triangle_plot=False):
        '''
        Generate the relevant plots once the sampling is done.
        '''
        samples = self.flatchain

        plot_walkers(self.outdir + self.fname + "_chain_pos.png", samples, labels=self.param_tuple)

        if triangle_plot:
            import triangle
            figure = triangle.corner(samples, labels=self.param_tuple, quantiles=[0.16, 0.5, 0.84],
                                     show_titles=True, title_args={"fontsize": 12})
            figure.savefig(self.outdir + self.fname + "_triangle.png")

            plt.close(figure)

class StellarSampler(PSampler):
    """
    Subclasses the Sampler specifically for stellar parameters



    """
    def __init__(self, **kwargs):
        '''
        :param pconns: Collection of parent ends of the PIPEs
        :type pconns: dict

        :param starting_param_dict:
            the dictionary of starting parameters

        :param cov:
            the MH proposal

        :param fix_logg:
            fix logg? If so, to what value?

        :param debug:

        :param args: []
        '''

        self.fix_logg = kwargs.get("fix_logg", False)
        starting_pram_dict = kwargs.get("starting_param_dict")
        self.param_tuple = self.startdict_to_tuple(starting_pram_dict)
        print("param_tuple is {}".format(self.param_tuple))
        self.p0 = np.array([starting_pram_dict[key] for key in self.param_tuple])

        kwargs.update({"p0":self.p0, "revertfn":self.revertfn, "acceptfn": self.acceptfn, "lnprobfn":self.lnprob})
        super(StellarSampler, self).__init__(**kwargs)

        #self.pconns is a dictionary of parent connections to each PIPE connecting to the child processes.
        self.spectrum_ids = sorted(self.pconns.keys())
        self.fname = "stellar"

    def startdict_to_tuple(self, startdict):
        tup = ()
        for param in C.stellar_parameters:
            #check if param is in keys, if so, add to the tuple
            if param in startdict:
                tup += (param,)
        return tup

    def reset(self):
        super(StellarSampler, self).reset()

    def revertfn(self):
        '''
        Revert the model to the previous state of parameters, in the case of a rejected MH proposal.
        '''
        self.logger.debug("reverting stellar parameters")
        self.prior = self.prior_last

        #Decide we don't want these stellar params. Tell the children to reject the proposal.
        for pconn in self.pconns.values():
            pconn.send(("DECIDE", False))

    def acceptfn(self):
        '''
        Execute this if the MH proposal is accepted.
        '''
        self.logger.debug("accepting stellar parameters")
        #Decide we do want to keep these stellar params. Tell the children to accept the proposal.
        for pconn in self.pconns.values():
            pconn.send(("DECIDE", True))

    def lnprob(self, p):
        # We want to send the same stellar parameters to each model,
        # but also send the different vz and logOmega parameters
        # to the separate spectra, based upon spectrum_id.
        #self.logger.debug("StellarSampler lnprob p is {}".format(p))

        #Extract only the temp, logg, Z, vsini parameters
        if not self.fix_logg:
            params = self.zip_p(p[:4])
            others = p[4:]
        else:
            #Coming in as temp, Z, vsini, vz, logOmega...
            params = self.zip_p(p[:3])
            others = p[3:]
            params.update({"logg": self.fix_logg})

        # Prior
        self.prior_last = self.prior

        logg = params["logg"]
        self.prior = -0.5 * (logg - 5.0)**2/(0.05)**2

        #others should now be either [vz, logOmega] or [vz0, logOmega0, vz1, logOmega1, ...] etc. Always div by 2.
        #split p up into [vz, logOmega], [vz, logOmega] pairs that update the other parameters.
        #mparams is now a list of parameter dictionaries

        #Now, pack up mparams into a dictionary to send the right stellar parameters to the right subprocesses
        mparams = {}
        for (spectrum_id, order_id), (vz, logOmega) in zip(self.spectrum_ids, grouper(others, 2)):
            p = params.copy()
            p.update({"vz":vz, "logOmega":logOmega})
            mparams[spectrum_id] = p

        self.logger.debug("updated lnprob params: {}".format(mparams))

        lnps = np.empty((self.nprocs,))

        #Distribute the calculation to each process
        self.logger.debug("Distributing params to children")
        for ((spectrum_id, order_id), pconn) in self.pconns.items():
            #Parse the parameters into what needs to be sent to each Model here.
            pconn.send(("LNPROB", mparams[spectrum_id]))

        #Collect the answer from each process
        self.logger.debug("Collecting params from children")
        for i, pconn in enumerate(self.pconns.values()):
            lnps[i] = pconn.recv()

        self.logger.debug("lnps : {}".format(lnps))
        s = np.sum(lnps)
        self.logger.debug("sum lnps {}".format(s))
        return s + self.prior

class NuisanceSampler(Sampler):
    def __init__(self, **kwargs):
        '''

        :param OrderModel: the parallel.OrderModel instance

        :param starting_param_dict: the dictionary of starting parameters

        :param cov:
            the MH proposal

        :param debug:

        :param args: []

        '''

        starting_param_dict = kwargs.get("starting_param_dict")
        self.param_tuple = self.startdict_to_tuple(starting_param_dict)
        print("param_tuple is {}".format(self.param_tuple))
        #print("param_tuple length {}".format(len(self.param_tuple)))

        chebs = [starting_param_dict["cheb"][key] for key in self.cheb_tup]
        covs = [starting_param_dict["cov"][key] for key in self.cov_tup]
        regions = starting_param_dict["regions"]
        #print("initializing {}".format(regions))
        regs = [regions[id][kk] for id in sorted(regions) for kk in C.cov_region_parameters]
        #print("regs {}".format(regs))

        self.p0 = np.array(chebs + covs + regs)

        kwargs.update({"p0":self.p0, "revertfn":self.revertfn, "lnprobfn":self.lnprob})
        super(NuisanceSampler, self).__init__(**kwargs)

        self.model = kwargs.get("OrderModel")
        spectrum_id, order_id = self.model.id
        order = kwargs.get("order", order_id)
        self.fname = "{}/{}/{}".format(spectrum_id, order, "nuisance")
        self.params = None
        self.prior_params = kwargs.get("prior_params", None)
        if self.prior_params:
            self.sigma0 = self.prior_params["regions"]["sigma0"]
            self.mus = self.prior_params["regions"]["mus"]
            self.mu_width = self.prior_params["regions"]["mu_width"]
            self.sigma_knee = self.prior_params["regions"]["sigma_knee"]
            self.frac_global = self.prior_params["regions"]["frac_global"]

    def startdict_to_tuple(self, startdict):
        #This is a little more tricky than the stellar parameters.
        #How are the keys stored and passed in the dictionary?
        #{"cheb": [c0, c1, c2, ..., cn], "cov": [sigAmp, logAmp, l],
        #        "regions":{0: [logAmp, ], 1: [], N:[] }}

        #Serialize the cheb parameters
        self.ncheb = len(startdict["cheb"])
        self.cheb_tup = ("logc0",) + tuple(["c{}".format(i) for i in range(1, self.ncheb)])

        #Serialize the covariance parameters
        self.ncov = 3
        cov_tup = ()
        for param in C.cov_global_parameters:
            #check if param is in keys, if so, add to the tuple
            if param in startdict["cov"]:
                cov_tup += (param,)
        self.cov_tup = cov_tup

        regions_tup = ()
        self.regions = startdict.get("regions", None)
        if self.regions:
            self.nregions = len(self.regions)
            for key in sorted(self.regions.keys()):
                for kk in C.cov_region_parameters:
                    regions_tup += ("r{:0>2}-{}".format(key,kk),)
            self.regions_tup = regions_tup
        else:
            self.nregions = 0
            self.regions_tup = ()


        tup = self.cheb_tup + self.cov_tup + self.regions_tup
        #This should look like
        #tup = ("c0", "c1", ..., "cn", "sigAmp", "logAmp", "l", "r00_logAmp", "r00_mu", "r00_sigma",
        # "r01_logAmp", ..., "rNN_sigma")
        return tup

    def zip_p(self, p):
        '''
        Convert the vector to a dictionary
        '''
        cheb = dict(zip(self.cheb_tup, p[:self.ncheb]))
        cov = dict(zip(self.cov_tup, p[self.ncheb:self.ncheb+self.ncov]))
        regions = p[-self.nregions*3:]
        rdict = {}
        for i in range(self.nregions):
            rdict[i] = dict(zip(("logAmp", "mu", "sigma"), regions[i*3:3*(i+1)]))

        params = {"cheb":cheb, "cov":cov, "regions":rdict}
        return params

    def revertfn(self):
        self.logger.debug("reverting model")
        self.model.prior = self.prior_last
        self.params = self.params_last
        self.model.revert_nuisance()

    def lnprob(self, p):
        self.params_last = self.params
        params = self.zip_p(p)
        self.params = params
        self.logger.debug("Updating nuisance params {}".format(params))

        # Nuisance parameter priors implemented here
        self.prior_last = self.model.prior
        # Region parameter priors implemented here
        if self.nregions > 0:
            regions = params["regions"]
            keys = sorted(regions)

            #Unpack the region parameters into a vector of mus, amps, and sigmas
            amps = 10**np.array([regions[key]["logAmp"] for key in keys])
            cov_amp = 10**params["cov"]["logAmp"]

            #First check to make sure that amplitude can't be some factor less than the global covariance
            if np.any(amps < (cov_amp * self.frac_global)):
                return -np.inf

            mus = np.array([regions[key]["mu"] for key in keys])
            sigmas = np.array([regions[key]["sigma"] for key in keys])

            #Make sure the region hasn't strayed too far from the original specification
            if np.any(np.abs(mus - self.mus) > self.sigma0):
                # The region has strayed too far from the original specification
                return -np.inf

            #Use a Gaussian prior on mu, that it keeps the region within the original setting.
            # 1/(sqrt(2pi) * sigma) exp(-0.5 (mu-x)^2/sigma^2)
            #-ln(sigma * sqrt(2 pi)) - 0.5 (mu - x)^2 / sigma^2
            #width = 0.04
            lnGauss = -0.5 * np.sum(np.abs(mus - self.mus)**2/self.mu_width**2 -
                                    np.log(self.mu_width * np.sqrt(2. * np.pi)))

            # Use a ln(logistic) function on sigma, that is flat before the knee and dies off for anything
            # greater, to prevent dilution into global cov kernel
            lnLogistic = np.sum(np.log(-1./(1. + np.exp(self.sigma_knee - sigmas)) + 1.))

            self.model.prior = lnLogistic + lnGauss

        try:
            self.model.update_nuisance(params)
            lnp = self.model.evaluate() # also sets OrderModel.lnprob to proposed value. Includes self.model.prior
            return lnp
        except C.ModelError:
            return -np.inf

def main():
    print("Starting main of model")

    pass

if __name__ == "__main__":
    main()
