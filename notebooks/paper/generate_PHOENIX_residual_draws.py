import numpy as np
from StellarSpectra.model import Model
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
from StellarSpectra import utils

myDataSpectrum = DataSpectrum.open("../../data/WASP14/WASP14-2009-06-14.hdf5", orders=np.array([21,22,23]))

myInstrument = TRES()

myHDF5Interface = HDF5Interface("../../libraries/PHOENIX_TRES_F.hdf5")

#Load a model using the JSON file
myModel = Model.from_json("WASP14_PHOENIX_model0_final.json", myDataSpectrum, myInstrument, myHDF5Interface)

myOrderModel = myModel.OrderModels[1]
model_flux = myOrderModel.get_spectrum()

spec = myModel.get_data()
wl = spec.wls[1]
fl = spec.fls[1]

model_fl = myOrderModel.get_spectrum()
residuals = fl - model_fl

mask = spec.masks[1]
cov = myModel.OrderModels[1].get_Cov().todense()

np.save("PHOENIX_covariance_matrix.npy", cov)

import sys; sys.exit()

draws = utils.random_draws(cov, num=50)

np.save("PHOENIX_residual_draws.npy", draws)