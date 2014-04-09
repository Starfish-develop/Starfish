import pytest
import numpy as np
import StellarSpectra.constants as C
from StellarSpectra.grid_tools import TRES, HDF5Interface
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.model import Model, StellarSampler, ChebSampler, CovSampler, MegaSampler

#temp: 6100
#logg: 4.0
#Z: -0.5
#vsini: 6
#vz: 13.7
#log_Omega: -19.7
#alpha: 0.2

class TestModel:
    def setup_class(self):
        myDataSpectrum = DataSpectrum.open("../data/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([22]))
        myInstrument = TRES()
        myHDF5Interface = HDF5Interface("../libraries/PHOENIX_submaster.hdf5")

        stellar_Starting = {"temp":(6000, 6100), "logg":(3.9, 4.2), "Z":(-0.6, -0.3), "vsini":(4, 6), "vz":(15.0, 16.0), "logOmega":(-19.665, -19.664)}
        stellar_tuple = C.dictkeys_to_tuple(stellar_Starting)
        cheb_Starting = {"logc0": (-0.02, 0.02), "c1": (-.02, 0.02), "c2": (-0.02, 0.02), "c3": (-.02, 0.02)}
        cov_Starting = {"sigAmp":(0.9, 1.1), "logAmp":(-14.4, -14), "l":(0.1, 0.25)}
        cov_tuple = C.dictkeys_to_covtuple(cov_Starting)

        self.Model = Model(myDataSpectrum, myInstrument, myHDF5Interface, stellar_tuple=stellar_tuple, cov_tuple=cov_tuple)

    def test_update(self):
        p = np.array([6050., 3.95, -0.4, 5.0, 15.5, -19.665])
        params = self.Model.zip_stellar_p(p)
        self.Model.update_Model(params) #This also updates downsampled_fls
        #For order in myModel, do evaluate, and sum the results.

    def test_evaluate(self):
        p = np.array([6050., 3.95, -0.4, 5.0, 15.5, -19.665])
        params = self.Model.zip_stellar_p(p)
        self.Model.update_Model(params) #This also updates downsampled_fls

        #This is giving us problems
        self.Model.evaluate()

    def test_error(self):
        pass
#
# class TestOrderModel:
#     def setup_class(self):
#         myDataSpectrum = DataSpectrum.open("../data/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([22]))
#         myInstrument = TRES()
#         myHDF5Interface = HDF5Interface("../libraries/PHOENIX_submaster.hdf5")
#         self.orderModel = OrderModel(ModelSpectrum, DataSpectrum, index)
