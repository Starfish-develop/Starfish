import os

import numpy as np
import h5py

import Starfish
from Starfish.covariance import Sigma, V12, V22, V12m, V22m
from Starfish import constants as C
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

        self.V11 = self.iPhiPhi + Sigma(self.pca.gparams, self.h2params)

        self._params = None  # Where we want to interpolate

        self.V12 = None
        self.V22 = None
        self.mu = None
        self.sig = None

    @classmethod
    def open(cls, filename=Starfish.PCA["path"]):
        '''
        Create an Emulator object from an HDF5 file.
        '''
        # Create the PCAGrid from this filename
        filename = os.path.expandvars(filename)
        pcagrid = PCAGrid.open(filename)
        hdf5 = h5py.File(filename, "r")

        eparams = hdf5["eparams"][:]
        hdf5.close()
        return cls(pcagrid, eparams)

    def determine_chunk_log(self, wl_data):
        '''
        Possibly truncate the wl grid in response to some data. Also truncate eigenspectra, and flux_mean and flux_std.
        '''
        self.pca.determine_chunk_log(wl_data)
        self.wl = self.pca.wl

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, pars):

        # If the pars is outside of the range of emulator values, raise a ModelError
        if np.any(pars < self.min_params) or np.any(pars > self.max_params):
            raise C.ModelError("Querying emulator outside of original PCA parameter range.")

        # Assumes pars is a single parameter combination, as a 1D np.array
        self._params = pars

        # Do this according to R&W eqn 2.18, 2.19

        # Recalculate V12, V21, and V22.
        self.V12 = V12(self._params, self.pca.gparams, self.h2params, self.pca.m)
        self.V22 = V22(self._params, self.h2params, self.pca.m)

        # Recalculate the covariance
        self.mu = self.V12.T.dot(np.linalg.solve(self.V11, self.pca.w_hat))
        self.mu.shape = (-1)
        self.sig = self.V22 - self.V12.T.dot(np.linalg.solve(self.V11, self.V12))

    @property
    def matrix(self):
        return (self.mu, self.sig)

    def draw_many_weights(self, params):
        '''
        :param params: multiple parameters to produce weight draws at.
        :type params: 2D np.array
        '''

        # Local variables, different from instance attributes
        v12 = V12m(params, self.pca.gparams, self.h2params, self.pca.m)
        v22 = V22m(params, self.h2params, self.pca.m)

        mu = v12.T.dot(np.linalg.solve(self.V11, self.pca.w_hat))
        sig = v22 - v12.T.dot(np.linalg.solve(self.V11, v12))

        weights = np.random.multivariate_normal(mu, sig)

        # Reshape these weights into a 2D matrix
        weights.shape = (len(params), self.pca.m)

        return weights

    def draw_weights(self):
        '''
        Using the current settings, draw a sample of PCA weights
        '''

        if self.V12 is None:
            print("No parameters are set, yet. Must set parameters first.")
            return

        return np.random.multivariate_normal(self.mu, self.sig)

    def reconstruct(self):
        '''
        Reconstructing a spectrum using a random draw of weights. In this case,
        we are making the assumption that we are always drawing a weight at a
        single stellar value.
        '''

        weights = self.draw_weights()
        return self.pca.reconstruct(weights)


