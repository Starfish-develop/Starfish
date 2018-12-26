import multiprocessing as mp

import numpy as np
import pytest

from Starfish.grid_tools import chunk_list, air_to_vacuum, vacuum_to_air, vacuum_to_air_SLOAN, idl_float


class TestChunking:

    def test_chunk_list_shape(self, grid_points):
        chunked = chunk_list(grid_points)
        assert len(chunked) == mp.cpu_count()
        chunked = chunk_list(grid_points, 3)
        assert len(chunked) == 3


class TestWavelengthUtils:

    def test_air_to_vacuum(self):
        wavelengths = np.linspace(1e4, 5e4, 1000)
        outputs = air_to_vacuum(wavelengths)
        assert len(outputs) == len(wavelengths)

    def test_vacuum_to_air(self):
        wavelengths = np.linspace(1e4, 5e4, 1000)
        outputs = vacuum_to_air(wavelengths)
        assert len(outputs) == len(wavelengths)

    def test_vacuum_to_air_SLOAN(self):
        wavelengths = np.linspace(1e4, 5e4, 1000)
        outputs = vacuum_to_air_SLOAN(wavelengths)
        assert len(outputs) == len(wavelengths)

    @pytest.mark.parametrize('wavelengths', [
        np.linspace(1e4, 5e4, 1000),
        np.linspace(1e5, 5e5, 1000)
    ])
    def test_atv_vta_regression(self, wavelengths):
        np.testing.assert_array_almost_equal(wavelengths, vacuum_to_air(air_to_vacuum(wavelengths)), 2)

    @pytest.mark.parametrize('wavelengths', [
        np.linspace(1e4, 5e4, 1000),
        np.linspace(1e5, 5e5, 1000)
    ])
    def test_atv_vta_sloan_regression(self, wavelengths):
        np.testing.assert_array_almost_equal(wavelengths, vacuum_to_air_SLOAN(air_to_vacuum(wavelengths)), 0)


@pytest.mark.parametrize('idl, num', [
    ('1D4', 1e4),
    ('1.0', 1.0),
    ('1D-4', 1e-4),
    ('1d0', 1.0),
])
def test_idl_float(idl, num):
    np.testing.assert_almost_equal(idl_float(idl), num)
