from StellarSpectra.model import Model, StellarSampler
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
import StellarSpectra.constants as C
import numpy as np

myDataSpectrum = DataSpectrum.open("../data/FakeWASP14/WASP14", orders=np.array([22]))
myDataSpectrum.fls[0] = np.load("../data/FakeWASP14/fakeWASP.noisey.fl.npy")

#To change sigma to the correlated version
# myDataSpectrum.sigmas[0] = np.load("../data/FakeWASP/fakeWASP.sigma.npy")

myInstrument = TRES()
myHDF5Interface = HDF5Interface("../libraries/PHOENIX_submaster.hdf5")

stellar_Starting = {"temp":6000, "logg":3.9, "Z":-0.7, "vsini":5.0, "vz":15, "logOmega":-19.665}
#Note that these values are sigma^2!!
stellar_MH_cov = np.array([2, 0.02, 0.02, 0.02, 0.02, 5e-4])**2 * np.identity(len(stellar_Starting))
stellar_tuple = C.dictkeys_to_tuple(stellar_Starting)



#cheb_Starting = {"c1": -.017, "c2": -.017, "c3": -.003}
#Note that these values are sigma^2!!
#cheb_MH_cov = np.array([2e-3, 2e-3, 2e-3])**2 * np.identity(len(cheb_Starting))

#cov_Starting = {"sigAmp":1, "logAmp":-14.0, "l":0.15}
#Note, THESE VALUES ARE sigma^2!!
#Note that these values are sigma^2!!
#cov_MH_cov = np.array([0.02, 0.02, 0.005])**2 * np.identity(len(cheb_Starting))
cov_tuple = ("sigAmp", "logAmp", "l")
region_tuple = ("h", "loga", "mu", "sigma")
#region_MH_cov = np.array([0.05, 0.04, 0.02, 0.02])**2 * np.identity(len(region_tuple))


myModel = Model(myDataSpectrum, myInstrument, myHDF5Interface, stellar_tuple=stellar_tuple, cov_tuple=cov_tuple, region_tuple=region_tuple)
#myModel.OrderModels[0].update_Cheb(np.array([-0.017, -0.017, -0.003]))
#myModel.OrderModels[0].CovarianceMatrix.update_global(cov_Starting)


myStellarSampler = StellarSampler(myModel, stellar_MH_cov, stellar_Starting)
#myChebSampler = ChebSampler(myModel, cheb_MH_cov, cheb_Starting, order_index=0)
#myCovSampler = CovGlobalSampler(myModel, cov_MH_cov, cov_Starting, order_index=0)
#myRegionsSampler = RegionsSampler(myModel, region_MH_cov, order_index=0)

myStellarSampler.burn_in(2000)
myStellarSampler.reset()

myStellarSampler.run(2000)
myStellarSampler.plot()
myStellarSampler.acceptance_fraction()


