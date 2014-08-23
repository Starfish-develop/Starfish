import numpy as np
from StellarSpectra.model import Model
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import SPEX, HDF5Interface
from StellarSpectra import utils

myDataSpectrum = DataSpectrum.open("../../data/Gl51/Gl51RA.hdf5", orders=np.array([0]))
myInstrument = SPEX()
myHDF5Interface = HDF5Interface("../../libraries/PHOENIX_SPEX_M.hdf5")

#Load a model using the JSON file
#Taken from:
# /home/ian/Grad/Research/Disks/StellarSpectra/output/Gl51/PHOENIX/RA/region/logg/4_8sig/
myModel = Model.from_json("Gl51_model0_final.json", myDataSpectrum, myInstrument, myHDF5Interface)

myOrderModel = myModel.OrderModels[0]
model_flux = myOrderModel.get_spectrum()

spec = myModel.get_data()
wl = spec.wls[0]
fl = spec.fls[0]

model_fl = myOrderModel.get_spectrum()
residuals = fl - model_fl

mask = spec.masks[0]
cov = myModel.OrderModels[0].get_Cov().todense()

np.save("Gl51_covariance_matrix.npy", cov)
import sys
sys.exit()


cov = np.load("Gl51_covariance_matrix.npy")

draws = utils.random_draws(cov, num=200)

np.save("Gl51_residual_draws.npy", draws)