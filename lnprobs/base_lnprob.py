from StellarSpectra.model import Model, StellarSampler, ChebSampler, CovSampler, MegaSampler
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
import StellarSpectra.constants as C
import numpy as np
import sys
from emcee.utils import MPIPool

myDataSpectrum = DataSpectrum.open("/home/ian/Grad/Research/Disks/StellarSpectra/tests/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array(22]))
myInstrument = TRES()
myHDF5Interface = HDF5Interface("/home/ian/Grad/Research/Disks/StellarSpectra/libraries/PHOENIX_submaster.hdf5")

stellar_Starting = {"temp":(6000, 6100), "logg":(3.9, 4.2), "Z":(-0.6, -0.3), "vsini":(4, 6), "vz":(15.0, 16.0), "logOmega":(-19.665, -19.664)}

stellar_tuple = C.dictkeys_to_tuple(stellar_Starting)

#cheb_Starting = {"c1": (-.02, -0.015), "c2": (-.0195, -0.0165), "c3": (-.005, .0)}
cheb_Starting = {"logc0": (-0.02, 0.02), "c1": (-.02, 0.02), "c2": (-0.02, 0.02), "c3": (-.02, 0.02)}
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

def lnprob_Cheb(p, index):
    #Select the correct order of myModel
    model = myModel.OrderModels[index]
    model.update_Cheb(p)
    return model.evaluate()

def lnprob_Cov(p, index):
    params = myModel.zip_Cov_p(p)
    model = myModel.OrderModels[index]
    if params["l"] > 0.4:
        return -np.inf
    try:
        model.update_Cov(params)
        return model.evaluate()
    except C.ModelError:
        return -np.inf

#def lnprob_Cov_region(p, order, region_num):
#    #Defining order and region_num at initialization time allows this to query into the covariance matrix at this region
#
#    #params look like {h, a, mu, sigma}
#    params = myModel.zip_Cov_p(p)
#    if params["l"] > 0.4:
#        return -np.inf
#    try:
#        myModel[order].update_Cov(params)
#        return myModel.evaluate()
#    except C.ModelError:
#        return -np.inf

# pool = MPIPool()
# if not pool.is_master():
#     pool.wait()
#     sys.exit(0)

myStellarSampler = StellarSampler(lnprob_Model, stellar_Starting)

#MegaSampler is initialized with a list of OrderSamplers

#Each OrderSampler is also a MegaSampler object
# Cheb22 = ChebSampler(lnprob_Cheb, cheb_Starting, index=0, pool=pool, plot_name="plots/out_cheb22.png")
# Cov22 = CovSampler(lnprob_Cov, cov_Starting, index=0, pool=pool, plot_name="plots/out_cov22.png")
# Order22Sampler = MegaSampler(samplers=(Cheb22, Cov22), burnInCadence=(10, 0), cadence=(1, 1))

Cheb23 = ChebSampler(lnprob_Cheb, cheb_Starting, index=0, pool=pool, plot_name="plots/out_cheb23.png")
Cov23 = CovSampler(lnprob_Cov, cov_Starting, index=0, pool=pool, plot_name="plots/out_cov23.png")
Order23Sampler = MegaSampler(samplers=(Cheb23, Cov23), burnInCadence=(10, 0), cadence=(1, 1))

# Cheb24 = ChebSampler(lnprob_Cheb, cheb_Starting, index=2, pool=pool, plot_name="plots/out_cheb24.png")
# Cov24 = CovSampler(lnprob_Cov, cov_Starting, index=2, pool=pool, plot_name="plots/out_cov24.png")
# Order24Sampler = MegaSampler(samplers=(Cheb24, Cov24), burnInCadence=(10, 0), cadence=(1, 1))

myMegaSampler = MegaSampler(samplers=(myStellarSampler, Order23Sampler), burnInCadence=(5, 2), cadence=(2, 1))


myMegaSampler.burn_in(2)
myMegaSampler.reset()
myMegaSampler.run(2)
pool.close()
myMegaSampler.plot()
