import numpy as np
import pytest

from Starfish.utils import calculate_dv, calculate_dv_dict, create_log_lam_grid


@pytest.mark.parametrize("dv", [100, 1000, 10000])
def test_grid_dv_regression(dv):
    grid = create_log_lam_grid(dv, 1e4, 4e4)
    dv_ = calculate_dv(grid["wl"])
    dv_d = calculate_dv_dict(grid)
    assert np.isclose(dv_, dv_d)
    assert dv_ <= dv


def test_grid_keys():
    grid = create_log_lam_grid(1000, 3000, 3e4)
    assert "wl" in grid
    assert "CRVAL1" in grid
    assert "CDELT1" in grid
    assert "NAXIS1" in grid


def test_calculate_dv_types():
    wave = np.linspace(1e4, 4e4)
    dv_np = calculate_dv(wave)
    dv = calculate_dv(wave.tolist())
    assert np.isclose(dv, dv_np)
    assert dv > 0


@pytest.mark.parametrize("start,end", [(3e4, 3e4), (4e4, 3e4), (-1, 30), (-10, -2)])
def test_invalid_points_grid(start, end):
    with pytest.raises(ValueError):
        create_log_lam_grid(1000, start, end)
