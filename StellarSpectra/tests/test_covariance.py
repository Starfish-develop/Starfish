import pytest
import numpy as np
from StellarSpectra.spectrum import *
from StellarSpectra.covariance import * #this is our cython model
import StellarSpectra.constants as C

class TestCCovarianceMatrix:
    def setup_class(self):
        self.dataspectrum = DataSpectrum.open("tests/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([21,22,23]))
        self.CovarianceMatrix = CovarianceMatrix(self.dataspectrum, order_index=1)

    def test_init(self):
        print("Initialized OK")

    def test_update_global(self):
        self.CovarianceMatrix.update_global({"sigAmp":1, "logAmp":0, "l":1})
        #at this point, the logdet should be != 0.0
        print(self.CovarianceMatrix.return_logdet())

    def test_evaluate(self):
        flux = 1e-13 * np.ones((2298,))
        print(self.CovarianceMatrix.evaluate(flux))

    def test_create_region(self):
        #Test to update with a=0.0 to see what's going wrong once we add in a new region.
        self.CovarianceMatrix.create_region({"h":0.02, "a":0.0, "mu":5180., "sigma":0.02})
        print(self.CovarianceMatrix.print_all_regions())
        flux = 1e-13 * np.ones((2298,))
        print(self.CovarianceMatrix.evaluate(flux))

    # def test_update_region(self):
    #     self.CovarianceMatrix.update_region(region_index=0, params={"h":0.05, "a":1.e-13, "mu":5180., "sigma":0.05})
    #     print(self.CovarianceMatrix.return_logdet())
    #     flux = 1e-13 * np.ones((2298,))
    #     print(self.CovarianceMatrix.evaluate(flux))


    # def test_bad_update(self):
    #
    #     with pytest.raises(C.ModelError) as e:
    #         self.cov.update({"sigAmp":1, "logAmp":0, "l":-1})
    #     print(e.value)
    #
    # def test_update(self):
    #     self.cov.update({"sigAmp":1, "logAmp":0, "l":1})
    #
    # def test_evaluate(self):
    #     lnprob = self.cov.evaluate(self.dataspectrum.fls[1])
    #     print(lnprob)
    #
    # def test_one_order(self):
    #     dataspectrum = DataSpectrum.open("tests/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([22]))
    #     cov = CovarianceMatrix(dataspectrum, 0)
    #     cov.update({"sigAmp":1, "logAmp":0, "l":1})
    #     lnprob = cov.evaluate(self.dataspectrum.fls[0])
    #     print(lnprob)