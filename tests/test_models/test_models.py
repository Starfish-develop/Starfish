import os

import pytest
import numpy as np

from Starfish.models import SpectrumModel


class TestSpectrumModel:

    GP = [6000, 4.0, 0]

    def test_param_dict(self, mock_model):
        assert mock_model['grid_param:0'] == self.GP[0]
        assert mock_model['grid_param:1'] == self.GP[1]
        assert mock_model['grid_param:2'] == self.GP[2]
        assert mock_model['vz'] == 0
        assert mock_model['Av'] == 0
        assert mock_model['scale'] == -10
        assert mock_model['vsini'] == 30

    def test_add_bad_param(self, mock_model):
        with pytest.raises(ValueError):
            mock_model['garbage_key'] = -4

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
        mock_model.freeze('grid_param:2')
        params = mock_model.get_param_dict()
        assert 'grid_param:0' in params
        assert 'grid_param:1' in params
        assert 'grid_param:2' not in params

    def test_freeze_thaw(self, mock_model):
        pre = mock_model['grid_param:1']
        mock_model.freeze('grid_param:1')
        assert 'grid_param:1' not in mock_model.get_param_dict()
        mock_model.thaw('grid_param:1')
        assert 'grid_param:1' in mock_model.get_param_dict()
        assert mock_model.grid_params[1] == pre

    def test_get_set_param_dict(self, mock_model):
        P0 = mock_model.get_param_dict()
        mock_model.set_param_dict(P0)
        P1 = mock_model.get_param_dict()
        assert P0 == P1

    def test_set_param_dict_frozen_params(self, mock_model):
        P0 = mock_model.get_param_dict()
        mock_model.freeze('grid_param:2')
        P0['grid_param:2'] = 7
        mock_model.set_param_dict(P0)
        assert mock_model['grid_param:2'] == 0

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
        assert mock_model['grid_param:2'] == 7

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
