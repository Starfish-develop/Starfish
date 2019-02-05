import numpy as np
import pytest

from Starfish.models.parameters import SpectrumParameter


class TestSpectrumParameter:

    def test_become_ndarray(self, mock_parameter):
        assert isinstance(mock_parameter.grid_params, np.ndarray)
        assert isinstance(mock_parameter.cheb, np.ndarray)

    def test_cheb(self, mock_parameter):
        assert mock_parameter.cheb[0] == 1

    def test_to_array_and_back(self, mock_parameter):
        p0 = mock_parameter.to_array()
        assert p0.ndim == 1
        param = SpectrumParameter.from_array(p0, 3)
        assert mock_parameter == param

    def test_save_and_load(self, mock_parameter, tmpdir):
        name = tmpdir.join('spectrum.json')
        mock_parameter.save(name)
        param = SpectrumParameter.load(name)
        assert param == mock_parameter

    def test_not_equal(self, mock_parameter):
        other = SpectrumParameter(grid_params=[5400, 3.2, 1.0])
        assert not mock_parameter == other
