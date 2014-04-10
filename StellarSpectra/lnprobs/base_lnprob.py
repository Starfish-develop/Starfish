from StellarSpectra.model import Model, StellarSampler, ChebSampler, CovSampler, MegaSampler
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
import StellarSpectra.constants as C
import numpy as np
import sys
from emcee.utils import MPIPool

myDataSpectrum = DataSpectrum.open("../data/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([22]))
myInstrument = TRES()
myHDF5Interface = HDF5Interface("../libraries/PHOENIX_submaster.hdf5")

stellar_Starting = {"temp":(6000, 6100), "logg":(3.9, 4.2), "Z":(-0.6, -0.3), "vsini":(4, 6), "vz":(15.0, 16.0), "logOmega":(-19.665, -19.664)}

stellar_tuple = C.dictkeys_to_tuple(stellar_Starting)

#cheb_Starting = {"c1": (-.02, -0.015), "c2": (-.0195, -0.0165), "c3": (-.005, .0)}
cheb_Starting = {"c1": (-.02, 0.02), "c2": (-0.02, 0.02), "c3": (-.02, 0.02)}
cov_Starting = {"sigAmp":(0.9, 1.1), "logAmp":(-14.4, -14), "l":(0.1, 0.25)}
cov_tuple = C.dictkeys_to_covtuple(cov_Starting)

myModel = Model(myDataSpectrum, myInstrument, myHDF5Interface, stellar_tuple=stellar_tuple, cov_tuple=cov_tuple)

def lnprob_Model(p):
    params = myModel.zip_stellar_p(p)
    try:
        myModel.update_Model(params) #This also updates downsampled_fls
        #For order in myModel, do evaluate, and sum the results.

        return myModel.evaluate()
    except C.ModelError:
        return -np.inf

def lnprob_Cheb(p, order_index):
    #Select the correct order of myModel
    model = myModel.OrderModels[order_index]
    model.update_Cheb(p)
    return model.evaluate()

def lnprob_Cov(p, order_index):
    params = myModel.zip_Cov_p(p)
    model = myModel.OrderModels[order_index]
    if params["l"] > 0.4:
        return -np.inf
    try:
        model.update_Cov(params)
        return model.evaluate()
    except C.ModelError:
        return -np.inf

def lnprob_Cov_region(p, order_index, region_index):
    '''defining order and region_num at initialization time allows this to query into the covariance matrix at
    this region
    p looks like {h, a, mu, sigma}
    '''
    params = myModel.zip_Cov_p(p)
    if params["l"] > 0.4: #apply logic to make sure reasonable parameters
        return -np.inf
    try:
        myModel[order].update_Cov(params)
        return myModel.evaluate()
    except C.ModelError:
        return -np.inf

# pool = MPIPool()
# if not pool.is_master():
#     pool.wait()
#     sys.exit(0)

myStellarSampler = StellarSampler(lnprob_Model, stellar_Starting)

#MegaSampler is initialized with a list of OrderSamplers
#Each OrderSampler is also a MegaSampler object


Cheb23 = ChebSampler(lnprob_Cheb, cheb_Starting, index=0, plot_name="plots/out_cheb23.png")
Cov23 = CovSampler(lnprob_Cov, cov_Starting, index=0, plot_name="plots/out_cov23.png")
Order23Sampler = MegaSampler(samplers=(Cheb23, Cov23), burnInCadence=(3, 3), cadence=(3, 3))

myMegaSampler = MegaSampler(samplers=(myStellarSampler, Order23Sampler), burnInCadence=(5, 1), cadence=(5, 1))

myMegaSampler.burn_in(30)
myMegaSampler.reset()
myMegaSampler.run(40)
# pool.close()
myMegaSampler.plot()
