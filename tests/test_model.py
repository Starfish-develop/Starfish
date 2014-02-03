import pytest
from StellarSpectra.model import *

def test_create_log_lam_grid():
    print(create_log_lam_grid(min_WL=(0.08,5000)))
    print(create_log_lam_grid(min_V=2))
    print(create_log_lam_grid(min_V=8))
    with pytest.raises(ValueError) as e:
        create_log_lam_grid()
    print(e.value)

class TestBaseSpectrum:
    def setup_class(self):
        self.spec = BaseSpectrum(np.linspace(5000, 5100, num=3000), np.random.normal(size=(3000,)))

    def test_metadata(self):
        print(self.spec.metadata)
        self.spec.add_metadata(("hello","hi"))
        print(self.spec.metadata)

        anotherSpec = BaseSpectrum(np.linspace(5000, 5100, num=3000), np.random.normal(size=(3000,)))
        print(anotherSpec.metadata)
        anotherSpec.add_metadata(("hello","hi"))
        print(anotherSpec.metadata)

class TestBase1DSpectrum(TestBaseSpectrum):
    def setup_class(self):
        self.spec = Base1DSpectrum(np.linspace(5000, 5100, num=3000), np.random.normal(size=3000,))

    def test_calculate_log_lam_grid(self):
        log_lam_grid = self.spec.calculate_log_lam_grid()
        print(log_lam_grid)

class TestLogLambdaSpectrum:
    pass