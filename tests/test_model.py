import pytest
from StellarSpectra.spectrum import DataSpectrum
from StellarSpectra.grid_tools import TRES, HDF5Interface
from StellarSpectra.model import *
import StellarSpectra.constants as C
import numpy as np

#temp: 6100
#logg: 4.0
#Z: -0.5
#vsini: 6
#vz: 13.7
#log_Omega: -19.7
#alpha: 0.2

class TestModel:
    def setup_class(self):
        self.DataSpectrum = DataSpectrum.open("/home/ian/Grad/Research/Disks/StellarSpectra/tests/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([21,22,23]))
        self.Instrument = TRES()
        self.HDF5Interface = HDF5Interface("/home/ian/Grad/Research/Disks/StellarSpectra/libraries/PHOENIX_submaster.hdf5")
        self.model = Model(self.DataSpectrum, self.Instrument, self.HDF5Interface, ("temp", "logg", "Z", "alpha", "vsini", "vz", "Av", "logOmega"), cov_tuple=("sigAmp", "logAmp", "l"))

    def test_evaluate(self):
        print(self.model.evaluate())

    def test_update_model(self):
        self.model.update_Model({"temp":6102, "logg":3.9, "Z":-0.5, "alpha":0.2, "vsini":6, "vz":13.7, "Av":0.0, "logOmega":-19.7})
        print(self.model.evaluate())

    def test_no_alpha(self):
        model = Model(self.DataSpectrum, self.Instrument, self.HDF5Interface, ("temp", "logg", "Z", "vsini", "vz", "Av", "logOmega"), cov_tuple=("sigAmp", "logAmp", "l"))
        model.update_Model({"temp":6102, "logg":3.9, "Z":-0.5, "alpha":0.2, "vsini":6, "vz":13.7, "Av":0.0, "logOmega":-19.7})
        print(model.evaluate())


    def test_update_Cheb(self):
        self.model.update_Cheb(np.array([1, -0.015, -0.015, -0.006, 1, 0, 0, 0, 1, 0, 0, 0]))
        print(self.model.evaluate())

    def test_update_Cov(self):
        self.model.update_Cov({"sigAmp":1, "logAmp":1, "l":1})
        print(self.model.evaluate())

        self.model.update_Cov({"sigAmp":1, "logAmp":1e-10, "l":1})
        print(self.model.evaluate())
