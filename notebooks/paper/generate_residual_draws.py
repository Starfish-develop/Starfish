import numpy as np
# from StellarSpectra.model import Model
# from StellarSpectra.spectrum import DataSpectrum
# from StellarSpectra.grid_tools import TRES, SPEX, HDF5Interface
from StellarSpectra import utils

#
# myDataSpectrum = DataSpectrum.open("../../data/WASP14/WASP14-2009-06-14.hdf5", orders=np.array([21, 22, 23]))
# myInstrument = TRES()
# myHDF5Interface = HDF5Interface("../../libraries/Kurucz_TRES.hdf5")
#
# #Load a model using the JSON file
# #Taken from:
# #/n/home07/iczekala/StellarSpectra/output/WASP14/Kurucz/21_22_23/logg/cov/2014-08-06/run18
# myModel = Model.from_json("WASP14_Kurucz_logg_model_final.json", myDataSpectrum, myInstrument, myHDF5Interface)
#
# myOrderModel = myModel.OrderModels[1]
# model_flux = myOrderModel.get_spectrum()
#
# spec = myModel.get_data()
# wl = spec.wls[1]
# fl = spec.fls[1]
#
# model_fl = myOrderModel.get_spectrum()
# residuals = fl - model_fl
#
# mask = spec.masks[1]
# cov = myModel.OrderModels[1].get_Cov().todense()
#
# np.save("kurucz_covariance_matrix.npy", cov)
# import sys
# sys.exit()
cov = np.load("kurucz_covariance_matrix.npy")

draws = utils.random_draws(cov, num=200, nprocesses=50)

np.save("krucuz_residual_draws.npy", draws)