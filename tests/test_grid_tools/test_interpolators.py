import pytest

import numpy as np
import pytest

from Starfish.grid_tools import IndexInterpolator


class TestIndexInterpolator:

    @pytest.fixture
    def mock_index_interpolator(self, grid_points):
        yield IndexInterpolator(grid_points)

    @pytest.mark.parametrize('input, expected', [
        [(6150, 4.25, 0.0), [[(6100, 4.0, 0.0), (6200, 4.5, 0.0)], [(0.5, 0.5, 1), (0.5, 0.5, 0)]]],
        [(6200, 4.1, -0.2), [[(6200, 4.0, -0.5), (6200, 4.5, 0.0)], [(1, 0.8, 0.4), (0, 0.2, 0.6)]]],
        [(6100, 4.0, 0.0), [[(6100, 4.0, 0.0), (6100, 4.0, 0.0)], [(1, 1, 1), (0, 0, 0)]]],
    ])
    def test_weights(self, input, expected, mock_index_interpolator):
        output = mock_index_interpolator(input)
        np.testing.assert_array_almost_equal(output, expected, 4)

class TestInterpolator:

    @pytest.fixture
    def mock_interpolator(self, mock_hdf5_interface):
        pass