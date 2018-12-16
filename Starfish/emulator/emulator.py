import os

import numpy as np
from Starfish.covariance import Sigma, V12, V22, V12m, V22m

import Starfish.constants as C
from Starfish import config
from .pca import PCAGrid, skinny_kron


class Emulator:
    def __init__(self, pca, eparams):
        """
        Provide the emulation products.

        :param pca: object storing the principal components, the eigenpsectra
        :type pca: PCAGrid
        :param eparams: Optimized GP hyperparameters.
        :type eparams: 1D np.array
        """

        self.pca = pca
        self.lambda_xi = eparams[0]
        self.h2params = eparams[1:].reshape(self.pca.m, -1) ** 2

        # Determine the minimum and maximum bounds of the grid
        self.min_params = np.min(self.pca.gparams, axis=0)
        self.max_params = np.max(self.pca.gparams, axis=0)

        # self.eigenspectra = self.PCAGrid.eigenspectra
        self.dv = self.pca.dv
        self.wl = self.pca.wl
        self.iPhiPhi = (1. / self.lambda_xi) * np.linalg.inv(skinny_kron(self.pca.eigenspectra, self.pca.M))
        self.v11 = self.iPhiPhi + Sigma(self.pca.gparams, self.h2params)

    @classmethod
    def open(cls, filename=config.PCA["path"]):
        """
        Create an Emulator object from an HDF5 file.
        """
        # Create the PCAGrid from this filename
        filename = os.path.expandvars(filename)
        pca_grid = PCAGrid.open(filename)
        return cls(pca_grid, pca_grid.eparams)

    def get_matrix(self, params):
        """
        Gets the mu and cov matrix for a given set of params

        :param params: The parameters to sample at. Should have same length as ``grid["parname"]`` in ``config.yaml``
        :type: iterable
        :return: ``tuple`` of mu, cov
        """
        params = np.array(params)
        # If the pars is outside of the range of emulator values, raise a ModelError
        if np.any(params < self.min_params) or np.any(params > self.max_params):
            raise C.ModelError("Querying emulator outside of original PCA parameter range.")

        # Do this according to R&W eqn 2.18, 2.19
        # Recalculate V12, V21, and V22.
        v12 = V12(params, self.pca.gparams, self.h2params, self.pca.m)
        v22 = V22(params, self.h2params, self.pca.m)

        # Recalculate the covariance
        mu = v12.T.dot(np.linalg.solve(self.v11, self.pca.w_hat)).squeeze()
        sig = v22 - v12.T.dot(np.linalg.solve(self.v11, v12))
        return mu, sig

    def load_flux(self, params, return_cov=False):
        """
        Interpolate a model given any parameters within the grid's parameter range using eigenspectrum reconstruction
        by sampling from the weight distributions.

        :param params: The parameters to sample at. Should have same length as ``grid["parname"]`` in ``config.yaml``
        :type: iterable
        :param return_cov: If true, will return the covariance matrix for the weights
        :type return_cov: bool
        :return: NDarray if return_cov is False, otherwise tuple of (NDarray, NDarray)

        .. warning::
            When returning the emulator covariance matrix, this is a costly operation and will return a
            datastructure with (N_pix x N_pix) data points. For now, don't do it.
        """
        params = np.array(params)
        mu, sig = self.get_matrix(params)
        weights = self.draw_weights(mu, sig)
        if return_cov:
            X = self.pca.eigenspectra.T
            C = np.linalg.multi_dot([X, sig, X.T])
            return self.reconstruct(weights), C
        else:
            return self.reconstruct(weights)

    def determine_chunk_log(self, wl_data):
        """
        Possibly truncate the wl grid in response to some data. Also truncate eigenspectra, and flux_mean and flux_std.
        """
        self.pca.determine_chunk_log(wl_data)
        self.wl = self.pca.wl

    def draw_weights(self, mu, sig):

        return np.random.multivariate_normal(mu, sig)

    def draw_many_weights(self, params):
        """
        :param params: multiple parameters to produce weight draws at.
        :type params: 2D np.array
        """
        # Local variables, different from instance attributes
        v12 = V12m(params, self.pca.gparams, self.h2params, self.pca.m)
        v22 = V22m(params, self.h2params, self.pca.m)

        mu = v12.T.dot(np.linalg.solve(self.v11, self.pca.w_hat))
        sig = v22 - v12.T.dot(np.linalg.solve(self.v11, v12))

        weights = np.random.multivariate_normal(mu, sig)

        # Reshape these weights into a 2D matrix
        weights.shape = (len(params), self.pca.m)

        return weights

    def reconstruct(self, weights):
        """
        Reconstructing a spectrum using a random draw of weights. In this case,
        we are making the assumption that we are always drawing a weight at a
        single stellar value.
        """
        return self.pca.reconstruct(weights)