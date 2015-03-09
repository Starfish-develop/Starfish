import numpy as np
import Starfish
from . import constants as C
from .grid_tools import Interpolator
from .spectrum import ModelSpectrum
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

class ThetaParam:
    '''
    An object holding the collection of parameters shared between all orders.

    :param grid: parameters corresponding to the dimensions of the grid.
    :type grid: 1D np.array
    '''
    def __init__(self, grid, vz=0.0, vsini=0.0, logOmega=0.0, Av=0.0):
        self.grid = grid
        self.vz = vz
        self.vsini = vsini
        self.logOmega = logOmega
        self.Av = Av

    def save(self, fname="theta.json"):
        '''
        Save the parameters to a JSON file
        '''
        f = open(fname, 'w')
        json.dump(self, f, cls=ThetaEncoder, indent=2, sort_keys=True)
        f.close()

    @classmethod
    def load(cls, fname="theta.json"):
        '''
        Load the parameters from a JSON file
        '''
        f = open(fname, "r")
        read = json.load(f) # read is a dictionary
        f.close()
        read["grid"] = np.array(read["grid"])
        return cls(**read)

class ThetaEncoder(json.JSONEncoder):
    '''
    Serialize an instance of o=ThetaParam() to JSON
    '''
    def default(self, o):
        try:
            mydict = {"grid":o.grid.tolist(),
                "vz":o.vz,
                "vsini":o.vsini,
                "logOmega":o.logOmega,
                "Av":o.Av}
        except TypeError:
            pass
        else:
            return mydict
        # Let the base class default method raise the TypeError, if there is one
        return json.JSONEncoder.default(self, o)

class PhiParam:
    '''
    An object holding the collection of parameters specific to a single order.
    '''
    def __init__(self, spec, order, cheb=np.zeros((Starfish.config["cheb_degree"],)),
        sigAmp=1.0, logAmp=0.0, l=10.0, regions=None):
        self.spec = spec
        self.order = order
        self.cheb = cheb
        self.sigAmp = sigAmp
        self.logAmp = logAmp
        self.l = l
        self.regions = regions

    def save(self, fname="phi.json"):
        f = open(Starfish.specfmt.format(self.spec, self.order) + fname, 'w')
        json.dump(self, f, cls=PhiEncoder, indent=2, sort_keys=True)
        f.close()

    @classmethod
    def load(cls, fname):
        '''
        Load the parameters from a JSON file
        '''
        f = open(fname, "r")
        read = json.load(f) # read is a dictionary
        f.close()
        read["cheb"] = np.array(read["cheb"])

        # Try to read regions
        if "regions" in read:
            read["regions"] = np.array(read["regions"])
        else:
            read["regions"] = None
        return cls(**read)


class PhiEncoder(json.JSONEncoder):
    '''
    Serialize an instance of o=PhiParam() to JSON
    '''
    def default(self, o):
        try:
            mydict = {"spec":o.spec, "order": o.order,
                "cheb":o.cheb.tolist(),
                "sigAmp":o.sigAmp,
                "logAmp":o.logAmp,
                "l":o.l}
            if o.regions is not None:
                mydict["regions"] = o.regions.tolist()
        except TypeError:
            pass
        else:
            return mydict
        # Let the base class default method raise the TypeError, if there is one
        return json.JSONEncoder.default(self, o)


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
