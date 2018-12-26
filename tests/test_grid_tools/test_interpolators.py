import numpy as np
import pytest

from Starfish.grid_tools import IndexInterpolator, Interpolator


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

    @pytest.mark.parametrize('input', [
        (1, 2),
        (6100, 4.2, -0.2, 0.0)
    ])
    def test_bounds_failure(self, input, mock_index_interpolator):
        with pytest.raises(ValueError):
            mock_index_interpolator(input)


class TestInterpolator:

    @pytest.fixture
    def mock_interpolator(self, mock_hdf5_interface):
        yield Interpolator(interface=mock_hdf5_interface)

    def test_simple_regression(self, mock_interpolator):
        params = (6150, 4.35, 0)
        np.testing.assert_array_equal(mock_interpolator(params), mock_interpolator.interpolate(params))

    @pytest.mark.parametrize('input', [
        (1, 2),
        (6100, 4.2, -0.2, 0.0)
    ])
    def test_bounds_failure(self, input, mock_interpolator):
        with pytest.raises(ValueError):
            mock_interpolator(input)

    def test_interpolation(self, mock_interpolator):
        output = mock_interpolator((6150, 4.35, -0.05))
        assert isinstance(output, np.ndarray)
        assert len(output) == len(mock_interpolator.interface.wl)

    def test_interpolation_on_grid(self, mock_interpolator, grid_points):
        for point in grid_points:
            original = mock_interpolator.interface.load_flux(point, header=False)
            interp = mock_interpolator(point)
            np.testing.assert_array_almost_equal(interp, original)
