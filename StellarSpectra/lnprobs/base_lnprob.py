from StellarSpectra.model import Model, StellarSampler, ChebSampler, CovGlobalSampler, RegionsSampler, MegaSampler
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
import StellarSpectra.constants as C
import numpy as np

myDataSpectrum = DataSpectrum.open("../data/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([22]))
myInstrument = TRES()
myHDF5Interface = HDF5Interface("../libraries/PHOENIX_submaster.hdf5")

stellar_Starting = {"temp":6000, "logg":4.05, "Z":-0.4, "vsini":10.5, "vz":15.5, "logOmega":-19.665}
#Note that these values are sigma^2!!
stellar_MH_cov = np.array([2, 0.02, 0.02, 0.02, 0.02, 5e-4])**2 * np.identity(len(stellar_Starting))

#Attempt at updating specific correlations
# #Temp/Logg correlation
# temp_logg = 0.2 * np.sqrt(0.01 * 0.001)
# stellar_MH_cov[0, 1] = temp_logg
# stellar_MH_cov[1, 0] = temp_logg
#
# #Temp/logOmega correlation
temp_logOmega = - 0.9 * np.sqrt(stellar_MH_cov[0,0] * stellar_MH_cov[5,5])
stellar_MH_cov[0, 5] = temp_logOmega
stellar_MH_cov[5, 0] = temp_logOmega

#We could make a function which takes the two positions of the parameters (0, 5) and then updates the covariance
#based upon a rho we feed it.

#We could test to see if these jumps are being executed in the right direction by checking to see what the 2D pairwise
# chain positions look like



stellar_tuple = C.dictkeys_to_tuple(stellar_Starting)

#
cheb_Starting = {"c1": -.017, "c2": -.017, "c3": -.003}
#Note that these values are sigma^2!!
cheb_MH_cov = np.array([2e-3, 2e-3, 2e-3])**2 * np.identity(len(cheb_Starting))

cov_Starting = {"sigAmp":1, "logAmp":-14.0, "l":0.15}
#Note, THESE VALUES ARE sigma^2!!
#Note that these values are sigma^2!!
cov_MH_cov = np.array([0.02, 0.02, 0.005])**2 * np.identity(len(cheb_Starting))
cov_tuple = C.dictkeys_to_cov_global_tuple(cov_Starting)

region_tuple = ("h", "loga", "mu", "sigma")
region_MH_cov = np.array([0.01, 0.02, 0.01, 0.01])**2 * np.identity(len(region_tuple))


myModel = Model(myDataSpectrum, myInstrument, myHDF5Interface, stellar_tuple=stellar_tuple, cov_tuple=cov_tuple, region_tuple=region_tuple)
myModel.OrderModels[0].update_Cheb(np.array([-0.017, -0.017, -0.003]))
myModel.OrderModels[0].CovarianceMatrix.update_global(cov_Starting)

# def lnprob_Model(p):
#     params = myModel.zip_stellar_p(p)
#     try:
#         myModel.update_Model(params) #This also updates downsampled_fls
#         #For order in myModel, do evaluate, and sum the results.
#
#         return myModel.evaluate()
#     except C.ModelError:
#         return -np.inf

myStellarSampler = StellarSampler(myModel, stellar_MH_cov, stellar_Starting)
myChebSampler = ChebSampler(myModel, cheb_MH_cov, cheb_Starting, order_index=0)
myCovSampler = CovGlobalSampler(myModel, cov_MH_cov, cov_Starting, order_index=0)
myRegionsSampler = RegionsSampler(myModel, region_MH_cov, order_index=0)

mySampler = MegaSampler(samplers=(myStellarSampler, myChebSampler, myCovSampler, myRegionsSampler), burnInCadence=(6, 6, 6, 2), cadence=(6, 6, 6, 2))
mySampler.burn_in(100)
mySampler.reset()

mySampler.run(100)
mySampler.plot()
mySampler.acceptance_fraction()


#my23RegionsSampler is initially empty of samplers, but has methods to create RegionSamplers as it goes


#my23OrderSampler = MegaSampler(samplers=(my23ChebSampler, my23CovSampler, my23RegionsSampler), ...)
#Final sampler will be
#mySampler = MegaSampler(samplers=(myStellarSampler, my22OrderSampler, my23OrderSampler, ...))

#


# print(myStellarSampler.sampler.flatchain)
# print(myStellarSampler.sampler.acceptance_fraction)
#
# def lnprob_Cheb(p, order_index):
#     #Select the correct order of myModel
#     model = myModel.OrderModels[order_index]
#     model.update_Cheb(p)
#     return model.evaluate()
#
# def lnprob_Cov(p, order_index):
#     params = myModel.zip_Cov_p(p)
#     model = myModel.OrderModels[order_index]
#     if params["l"] > 0.4:
#         return -np.inf
#     try:
#         model.update_Cov(params)
#         return model.evaluate()
#     except C.ModelError:
#         return -np.inf
#
# def lnprob_Cov_region(p, order_index, region_index):
#     '''defining order and region_num at initialization time allows this to query into the covariance matrix at
#     this region
#     p looks like {h, a, mu, sigma}
#     '''
#     params = myModel.zip_Cov_p(p)
#     if params["l"] > 0.4: #apply logic to make sure reasonable parameters
#         return -np.inf
#     try:
#         myModel[order_index].update_Cov(params)
#         return myModel.evaluate()
#     except C.ModelError:
#         return -np.inf
#
# def evaluate_region_logic(order_index):
#     '''
#     This is a method that RegionsSampler will call once self.logic_counter has overflown. It will check if there
#     are any residuals that exist beyond a certain threshold (3 x general structure?) that are not already covered by some line.
#
#     If such a region does exist, it will return a mu to initialize that line.
#
#     If none exist, it will return None.
#
#     Eventually add support to remove regions.
#
#     '''
#     model = myModel.OrderModels[order_index]
#     return model.evaluate_region_logic()
#
# # pool = MPIPool()
# # if not pool.is_master():
# #     pool.wait()
# #     sys.exit(0)
#
# myStellarSampler = StellarSampler(lnprob_Model, stellar_Starting)
#
# #MegaSampler is initialized with a list of OrderSamplers
# #Each OrderSampler is also a MegaSampler object
#
#
# Cheb23 = ChebSampler(lnprob_Cheb, cheb_Starting, index=0, plot_name="plots/out_cheb23.png")
# Cov23 = CovSampler(lnprob_Cov, cov_Starting, index=0, plot_name="plots/out_cov23.png")
#
# Regions23 = RegionsSampler()
#
# Order23Sampler = MegaSampler(samplers=(Cheb23, Cov23), burnInCadence=(3, 3), cadence=(3, 3))
#
# myMegaSampler = MegaSampler(samplers=(myStellarSampler, Order23Sampler), burnInCadence=(5, 1), cadence=(5, 1))
#
# myMegaSampler.burn_in(30)
# myMegaSampler.reset()
# myMegaSampler.run(40)
# # pool.close()
# myMegaSampler.plot()
