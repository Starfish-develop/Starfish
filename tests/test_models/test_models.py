import os
from collections import deque

import pytest
import numpy as np

from Starfish.models import SpectrumModel


class TestSpectrumModel:

    GP = [6000, 4.0, 0]

    def test_param_dict(self, mock_model):
        assert mock_model['T'] == self.GP[0]
        assert mock_model['logg'] == self.GP[1]
        assert mock_model['Z'] == self.GP[2]
        assert mock_model['vz'] == 0
        assert mock_model['Av'] == 0
        assert mock_model['log_scale'] == -10
        assert mock_model['vsini'] == 30

    def test_global_cov_param_dict(self, mock_model):
        assert mock_model.glob == mock_model['global']
        assert 'log_amp' in mock_model.glob
        assert 'log_ls' in mock_model.glob
        assert 'global:log_amp' in mock_model.get_param_dict(flat=True)

    def test_local_cov_param_dict(self, mock_model):
        assert mock_model.local == mock_model['local']
        assert len(mock_model.local) == 2
        assert mock_model.local[0]['mu'] == 1e4
        assert 'log_sigma' in mock_model.local[1]
        assert 'local:0:log_amp' in mock_model.get_param_dict(flat=True)
        assert 'local:1:mu' in mock_model.get_param_dict(flat=True)

    @pytest.mark.parametrize('param', [
        'global:log_amp',
        'local:0:log_amp'
    ])
    def test_cov_freeze(self, mock_model, param):
        assert param in mock_model.labels
        mock_model.freeze(param)
        assert param not in mock_model.labels
        mock_model.thaw(param)
        assert param in mock_model.labels


    def test_add_bad_param(self, mock_model):
        with pytest.raises(ValueError):
            mock_model['garbage_key'] = -4

    def test_labels(self, mock_model):
        assert mock_model.labels == list(mock_model.get_param_dict(flat=True))

    def test_grid_params(self, mock_model):
        assert np.all(mock_model.grid_params == self.GP)

    def test_transform(self, mock_model):
        flux, cov = mock_model()
        assert cov.shape == (len(flux), len(flux))
        assert flux.shape == mock_model.data.waves.shape

    def test_freeze_vsini(self, mock_model):
        mock_model.freeze('vsini')
        params = mock_model.get_param_dict()
        assert 'vsini' not in params

    def test_freeze_grid_param(self, mock_model):
        mock_model.freeze('logg')
        params = mock_model.get_param_dict()
        assert 'T' in params
        assert 'Z' in params
        assert 'logg' not in params

    def test_freeze_thaw(self, mock_model):
        pre = mock_model['logg']
        mock_model.freeze('logg')
        assert 'logg' not in mock_model.get_param_dict()
        mock_model.thaw('logg')
        assert 'logg' in mock_model.get_param_dict()
        assert mock_model.grid_params[1] == pre

    @pytest.mark.parametrize('flat', [
        False,
        True
    ])
    def test_get_set_param_dict(self, mock_model, flat):
        P0 = mock_model.get_param_dict(flat=flat)
        mock_model.set_param_dict(P0, flat=flat)
        P1 = mock_model.get_param_dict(flat=flat)
        assert P0 == P1

    def test_set_param_dict_frozen_params(self, mock_model):
        P0 = mock_model.get_param_dict()
        mock_model.freeze('Z')
        P0['Z'] = 7
        mock_model.set_param_dict(P0, flat=False)
        assert mock_model['Z'] == 0

    def test_get_set_parameters(self, mock_model):
        P0 = mock_model.get_param_vector()
        mock_model.set_param_vector(P0)
        P1 = mock_model.get_param_vector()
        assert np.all(P1 == P0)

    def test_set_wrong_length_param_vector(self, mock_model):
        P0 = mock_model.get_param_vector()
        P1 = np.append(P0, 7)
        with pytest.raises(ValueError):
            mock_model.set_param_vector(P1)

    def test_set_param_vector(self, mock_model):
        P0 = mock_model.get_param_vector()
        P0[2] = 7
        mock_model.set_param_vector(P0)
        assert mock_model['Z'] == 7

    def test_save_load(self, mock_model, tmpdir):
        path = os.path.join(tmpdir, 'model.json')
        P0 = mock_model.params
        P0_f = mock_model.get_param_dict()
        mock_model.save(path)
        mock_model.load(path)
        assert P0 == mock_model.params
        assert P0_f == mock_model.get_param_dict()

    def test_log_likelihood(self, mock_model):
        lnprob = mock_model.log_likelihood()
        assert np.isfinite(lnprob)

    def test_grad_log_likelihood_doesnt_exist(self, mock_model):
        with pytest.raises(NotImplementedError):
            mock_model.grad_log_likelihood()

    def test_str(self, mock_model):
        assert str(mock_model).startswith('SpectrumModel')

    def test_find_residuals_peaks(self, mock_model):
        n = 100
        fake_residual = np.random.randn(100, *mock_model.data.waves.shape)
        fake_residual[:, 10] = 1000
        mock_model.residuals = deque(fake_residual.tolist())
        peaks = mock_model.find_residual_peaks(num_residuals=50)
        assert mock_model.data.waves[10] == peaks[0]
        assert len(peaks) < len(mock_model.data.waves) - 2
