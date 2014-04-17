from StellarSpectra.model import Model, StellarSampler, ChebSampler, CovGlobalSampler, RegionsSampler, MegaSampler
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
import StellarSpectra.constants as C
import numpy as np

myDataSpectrum = DataSpectrum.open("data/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([22]))
myInstrument = TRES()
myHDF5Interface = HDF5Interface("libraries/PHOENIX_submaster.hdf5")

stellar_Starting = {"temp":6000, "logg":4.05, "Z":-0.4, "vsini":5.5, "vz":15.5, "logOmega":-19.665}
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
region_MH_cov = np.array([0.05, 0.04, 0.02, 0.02])**2 * np.identity(len(region_tuple))


myModel = Model(myDataSpectrum, myInstrument, myHDF5Interface, stellar_tuple=stellar_tuple, cov_tuple=cov_tuple, region_tuple=region_tuple)
myModel.OrderModels[0].update_Cheb(np.array([-0.017, -0.017, -0.003]))
myModel.OrderModels[0].CovarianceMatrix.update_global(cov_Starting)


myStellarSampler = StellarSampler(myModel, stellar_MH_cov, stellar_Starting)
myChebSampler = ChebSampler(myModel, cheb_MH_cov, cheb_Starting, order_index=0)
myCovSampler = CovGlobalSampler(myModel, cov_MH_cov, cov_Starting, order_index=0)
myRegionsSampler = RegionsSampler(myModel, region_MH_cov, order_index=0)

mySampler = MegaSampler(samplers=(myStellarSampler, myChebSampler, myCovSampler, myRegionsSampler), burnInCadence=(10, 6, 6, 2), cadence=(10, 6, 6, 2))
mySampler.burn_in(1000)
mySampler.reset()

mySampler.run(4000)
mySampler.plot()
mySampler.acceptance_fraction()


