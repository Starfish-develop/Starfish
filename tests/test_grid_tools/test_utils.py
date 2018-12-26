import multiprocessing as mp

import pytest
import numpy as np

from Starfish.grid_tools import chunk_list, determine_chunk_log, air_to_vacuum, vacuum_to_air, vacuum_to_air_SLOAN


class TestChunking:

    def test_chunk_list(self, grid_points):
        chunked = chunk_list(grid_points)
        assert chunked.shape[0] == mp.cpu_count()
        chunked = chunk_list(grid_points, 3)
        assert chunked.shape[0] == 3


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