import pytest
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
from StellarSpectra.model import *
import StellarSpectra.constants as C
import numpy as np

class TestModel:
    def setup_class(self):
        myDataSpectrum = DataSpectrum.open("/home/ian/Grad/Research/Disks/StellarSpectra/tests/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([21,22,23]))
        myInstrument = TRES()
        myHDF5Interface = HDF5Interface("/home/ian/Grad/Research/Disks/StellarSpectra/libraries/PHOENIX_submaster.hdf5")
        self.model = Model(myDataSpectrum, myInstrument, myHDF5Interface)

    def test_evaluate(self):
        print(self.model.evaluate())

    def test_update_model(self):
        self.model.update_Model({"temp":6002, "logg":3.9, "Z":-0.5, "alpha":0.2, "vsini":4, "vz":15, "Av":0.0, "Omega":2.2e-20})
        print(self.model.evaluate())

    def test_update_Cheb(self):
        self.model.update_Cheb(np.array([1, -0.015, -0.015, -0.006, 1, 0, 0, 0, 1, 0, 0, 0]))
        print(self.model.evaluate())



