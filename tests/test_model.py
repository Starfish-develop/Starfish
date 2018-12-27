import numpy as np
import pytest

from Starfish.model import ThetaParam, PhiParam


class TestThetaParam:

    @pytest.fixture(scope='class')
    def thetaparam(self):
        yield ThetaParam(grid=np.array([4000., 4.32, -0.2]),
                         vz=10., vsini=4.0, logOmega=-0.2, Av=0.3)

    @pytest.fixture
    def saved_file(self, thetaparam, tmpdir):
        outname = tmpdir.join("theta_test.json")
        thetaparam.save(fname=outname)
        yield outname

    def test_load(self, saved_file):
        loaded = ThetaParam.load(saved_file)
        np.testing.assert_array_equal(loaded.grid, np.array([4000., 4.32, -0.2]))
        assert loaded.vz == 10.
        assert loaded.vsini == 4.
        assert loaded.logOmega == -0.2
        assert loaded.Av == 0.3


class TestPhiParam:

    @pytest.fixture(scope='class')
    def phiparam(self):
        yield PhiParam(spectrum_id=0, order=22, fix_c0=True,
                       cheb=np.zeros((4,)), sigAmp=1.0, logAmp=-5.0, l=20., regions=np.ones((4, 3)))

    @pytest.fixture
    def saved_file(self, phiparam, tmpdir):
        yield phiparam.save(fname=tmpdir.join("phi_test.json"))

    def test_load(self, saved_file):
        spectrum_id = 0
        order = 22
        loaded = PhiParam.load(saved_file)
        assert loaded.spectrum_id == spectrum_id
        assert loaded.order == order
        assert loaded.fix_c0 == True
        np.testing.assert_array_equal(loaded.cheb, np.zeros((4,)))
        assert loaded.sigAmp == 1.0
        assert loaded.logAmp == -5.0
        assert loaded.l == 20.0
        np.testing.assert_array_equal(loaded.regions, np.ones((4, 3)))
