import numpy as np
import pytest

from Starfish.models.parameters import SpectrumParameter


class TestSpectrumParameter:

    @pytest.fixture(scope='class')
    def mock_param(self):
        yield SpectrumParameter(grid_params=[6000., 4.32, -0.2],
                                vz=10., vsini=4.0, logOmega=-0.2, Av=0.3, cheb=[1, ])

    def test_become_ndarray(self, mock_param):
        assert isinstance(mock_param.grid_params, np.ndarray)
        assert isinstance(mock_param.cheb, np.ndarray)

    def test_to_array_and_back(self, mock_param):
        p0 = mock_param.to_array()
        param = SpectrumParameter.from_array(p0)
        assert p0 == param

    def test_save_and_load(self, mock_param, tmpdir):
        name = tmpdir.join('spectrum.json')
        mock_param.save(name)
        param = SpectrumParameter.load(name)
        assert param == mock_param

    def test_not_equal(self, mock_param):
        other = SpectrumParameter(grid_params=[5400, 3.2, 1.0])
        assert not mock_param == other
