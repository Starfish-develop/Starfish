import numpy as np
from StellarSpectra.model import Model
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
from StellarSpectra import utils

myDataSpectrum = DataSpectrum.open("../../data/WASP14/WASP14-2009-06-14.hdf5", orders=np.array([22]))

myInstrument = TRES()

myHDF5Interface = HDF5Interface("../../libraries/PHOENIX_TRES_F.hdf5")

#Load a model using the JSON file
myModel = Model.from_json("../WASP14_22_model_final_region.json", myDataSpectrum, myInstrument, myHDF5Interface)

myOrderModel = myModel.OrderModels[0]
model_flux = myOrderModel.get_spectrum()

spec = myModel.get_data()
wl = spec.wls[0]
fl = spec.fls[0]

model_fl = myOrderModel.get_spectrum()
residuals = fl - model_fl

mask = spec.masks[0]
cov = myModel.OrderModels[0].get_Cov().todense()

draws = utils.random_draws(cov, num=50)

np.save("PHOENIX_residual_draws.npy", draws)