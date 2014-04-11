import pytest
import numpy as np
from StellarSpectra.spectrum import *
from StellarSpectra.covariance import * #this is our cython model
import StellarSpectra.constants as C

class TestCCovarianceMatrix:
    def setup_class(self):
        self.dataspectrum = DataSpectrum.open("tests/WASP14/WASP-14_2009-06-15_04h13m57s_cb.spec.flux", orders=np.array([21,22,23]))
        self.dataspectrum.fls = np.ones_like(self.dataspectrum.wls)
        print(self.dataspectrum.fls)
        self.flux = flux = np.zeros((2298,)) #so that the residuals == 1.0
        self.CovarianceMatrix = CovarianceMatrix(self.dataspectrum, order_index=1)

    def test_init(self):
        print("Initialized OK")

    def test_update_global(self):
        self.CovarianceMatrix.update_global({"sigAmp":1, "logAmp":0, "l":1})

    def test_evaluate(self):
        print("\n NO REGION lnprob = ", self.CovarianceMatrix.evaluate(self.flux))

    def test_create_region(self):
        self.CovarianceMatrix.create_region({"h":1.0, "a":1.0, "mu":5180., "sigma":1.0})
        print(self.CovarianceMatrix.print_all_regions())
        print("\n REGION lnprob = ", self.CovarianceMatrix.evaluate(self.flux))

    def test_two_regions_error(self):
        with pytest.raises(C.RegionError) as e:
            self.CovarianceMatrix.create_region({"h":1.0, "a":1.0, "mu":5130., "sigma":1.0})
        print(e.value)

    def test_add_two_regions(self):
        self.CovarianceMatrix.create_region({"h":1.0, "a":1.0, "mu":5200., "sigma":1.0})
        print(self.CovarianceMatrix.print_all_regions())
        print("\n TWO REGION lnprob = ", self.CovarianceMatrix.evaluate(self.flux))

    def test_update_first_region_error(self):
        with pytest.raises(C.RegionError) as e:
            self.CovarianceMatrix.update_region(0, {"h":1.0, "a":1.0, "mu":5185., "sigma":1.0})
        print(e.value)

    def test_update_first_region(self):
        self.CovarianceMatrix.update_region(0, {"h":1.0, "a":1.0, "mu":5182., "sigma":1.0})
        print(self.CovarianceMatrix.print_all_regions())
        print("\n updated TWO REGION lnprob = ", self.CovarianceMatrix.evaluate(self.flux))

    def test_update_second_region(self):
        self.CovarianceMatrix.update_region(1, {"h":1.0, "a":0.0, "mu":5200., "sigma":1.0})
        print(self.CovarianceMatrix.print_all_regions())
        print("\n updated TWO REGION lnprob = ", self.CovarianceMatrix.evaluate(self.flux))

#Create_region will really fuck up if mu < wl[0]. How to circumvent this?