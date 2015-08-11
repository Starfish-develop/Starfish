import pytest

import Starfish
from Starfish.model import ThetaParam, PhiParam
import numpy as np

class TestThetaParam:
    def setup_class(self):
        self.thetaparam = ThetaParam(grid=np.array([4000., 4.32, -0.2]),
        vz=10., vsini=4.0, logOmega=-0.2, Av=0.3)

    def test_save(self):
        self.thetaparam.save(fname="theta_test.json")

    def test_load(self):
        load = ThetaParam.load("theta_test.json")
        print(load.grid)
        print(load.vz)
        print(load.vsini)
        print(load.logOmega)
        print(load.Av)

class TestPhiParam:
    def setup_class(self):
        self.phiparam = PhiParam(spectrum_id=0, order=22, fix_c0=True,
        cheb=np.zeros((4,)), sigAmp=1.0, logAmp=-5.0, l=20., regions=np.ones((4, 3)))

    def test_save(self):
        self.phiparam.save(fname="phi_test.json")

    def test_load(self):
        load = PhiParam.load(Starfish.specfmt.format(0, 22) + "phi_test.json")
        print(load.spectrum_id)
        print(load.order)
        print(load.fix_c0)
        print(load.cheb)
        print(load.sigAmp)
        print(load.logAmp)
        print(load.l)
        print(load.regions)
