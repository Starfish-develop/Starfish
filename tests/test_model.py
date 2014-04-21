import pytest
import numpy as np
import StellarSpectra.constants as C
from StellarSpectra.grid_tools import TRES, HDF5Interface
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.model import Model



class TestModel:
    def setup_class(self):
        myDataSpectrum = DataSpectrum.open("../data/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([22]))
        myInstrument = TRES()
        myHDF5Interface = HDF5Interface("../libraries/PHOENIX_submaster.hdf5")

        stellar_Starting = {"temp":6000, "logg":4.05, "Z":-0.4, "vsini":10.5, "vz":15.5, "logOmega":-19.665}
        stellar_tuple = C.dictkeys_to_tuple(stellar_Starting)

        cheb_tuple = ("c1", "c2", "c3")
        cov_tuple = ("sigAmp", "logAmp", "l")
        region_tuple = ("h", "loga", "mu", "sigma")

        self.Model = Model(myDataSpectrum, myInstrument, myHDF5Interface, stellar_tuple=stellar_tuple, cheb_tuple=cheb_tuple,
                           cov_tuple=cov_tuple, region_tuple=region_tuple, outdir="")

    def test_update(self):
        self.Model.OrderModels[0].update_Cheb({"c1": -0.017, "c2": -0.017, "c3": -0.003})
        cov_Starting = {"sigAmp":1, "logAmp":-14.0, "l":0.15}
        self.Model.OrderModels[0].update_Cov(cov_Starting)

        params = {"temp":6005, "logg":4.05, "Z":-0.4, "vsini":10.5, "vz":15.5, "logOmega":-19.665}
        self.Model.update_Model(params) #This also updates downsampled_fls
        #For order in myModel, do evaluate, and sum the results.

    def test_evaluate(self):
        self.Model.evaluate()

    def test_to_json(self):
        self.Model.to_json()

