=================
Spectral Emulator
=================

.. py:module:: Starfish.emulator
   :synopsis: Return a probability distribution over possible interpolated spectra.

The spectral emulator can be likened to the engine behind *Starfish*. While the novelty of *Starfish* comes from using Gaussian processes to model and account for the covariances of spectral fits, we still need a way to produce model spectra by interpolating from our synthetic library. While we could interpolate spectra from the synthetic library using something like linear interpolation in each of the library parameters, it turns out that high signal-to-noise data requires something more sophisticated. This is because the error in any interpolation can constitute a significant portion of the error budget. This means that there is a chance that non-interpolated spectra (e.g., the parameters of the synthetic spectra in the library) might be given preference over any other interpolated spectra, and the posteriors will be peaked at the grid point locations. Because the spectral emulator returns a probability distribution over possible interpolated spectra, this interpolation error can be quantified and propagated forward into the likelihood calculation.

Eigenspectra decomposition
==========================

The first step of configuring the spectral emulator is to choose a subregion of the spectral library corresponding to the star that you will fit. Then, we want to decompose the information content in this subset of the spectral library into several *eigenspectra*. [Figure A.1 here].

The eigenspectra decomposition is performed via Principal Component Analysis (PCA). Thankfully, most of the heavy lifting is already implemented by the ``sklearn`` package.

:meth:`Emulator.from_grid` allows easy creation of spectral emulators from an :class:`Starfish.grid_tools.HDF5Interface`, which includes doing the initial PCA to create the eigenspectra.


.. code-block:: python

    >>> from Starfish.grid_tools import HDF5Interface
    >>> from Starfish.emulator import Emulator
    >>> emulator = Emulator.from_grid(HDF5Interface('grid.hdf5'))


Optimizing the emulator
=======================

Once the synthetic library is decomposed into a set of eigenspectra, the next step is to train the Gaussian Processes (GP) that will serve as interpolators. For more explanation about the choice of Gaussian Process covariance functions and the design of the emulator, see the appendix of our paper.

The optimization of the GP hyperparameters can be carried out by any maximum likelihood estimation framework, but we include a direct method that uses ``scipy.optimize.minimize``.

To optimize the code, we will use the :meth:`Emulator.train` routine.

Example optimizing using minimization optimizer

.. code-block:: python

    >>> from Starfish.grid_tools import HDF5Interface
    >>> from Starfish.emulator import Emulator
    >>> emulator = Emulator.from_grid(HDF5Interface('grid.hdf5'))
    >>> emulator
    Emulator
    --------
    Trained: False
    lambda_xi: 2.718
    Variances:
        10000.00
        10000.00
        10000.00
        10000.00
    Lengthscales:
        [ 600.00  1.50  1.50 ]
        [ 600.00  1.50  1.50 ]
        [ 600.00  1.50  1.50 ]
        [ 600.00  1.50  1.50 ]
    Log Likelihood: -1412.00
    >>> emulator.train()
    >>> emulator
    Emulator
    --------
    Trained: True
    lambda_xi: 2.722
    Variances:
        238363.85
        5618.02
        9358.09
        2853.22
    Lengthscales:
        [ 1582.39  3.19  3.11 ]
        [ 730.81  1.61  2.14 ]
        [ 1239.45  3.71  2.78 ]
        [ 1127.40  1.63  4.46 ]
    Log Likelihood: -1158.83
    >>> emulator.save('trained_emulator.hdf5')

.. note::

    The built in optimization target changes the state of the emulator, so even if the output of the minimizer has not converged, you can simply run :meth:`Emulator.train` again.

If you want to perform MLE with a different method, feel free to make use of the general modeling framework provided by the function :meth:`Emulator.get_param_vector`, :meth:`Emulator.set_param_vector`, and :meth:`Emulator.log_likelihood`.

Model spectrum reconstruction
=============================

Once the emulator has been optimized, we can finally use it as a means of interpolating spectra.

.. code-block:: python

    >>> from Starfish.emulator import Emulator
    >>> emulator = Emulator.load('trained_emulator.hdf5')
    >>> flux = emulator.load_flux([7054, 4.0324, 0.01])
    >>> wl = emu.wl

If you want to take advantage of the emulator covariance matrix, you must use the interface via the :meth:`Emulator.__call__` function

.. code-block:: python

    >>> from Starfish.emulator import Emulator
    >>> emulator = Emulator.load('trained_emulator.hdf5')
    >>> weights, cov = emulator([7054, 4.0324, 0.01])
    >>> X = emulator.eigenspectra * emulator.flux_std
    >>> flux = weights @ X + emulator.flux_mean
    >>> emu_cov = X.T @ weights @ X

Lastly, if you want to process the model, it is useful to process the eigenspectra before reconstructing, especially if a resampling action has to occur. The :class:`Emulator` provides the attribute :attr:`Emulator.bulk_fluxes` for such processing. For example

.. code-block:: python

    >>> from Starfish.emulator import Emulator
    >>> from Starfish.transforms import instrumental_broaden
    >>> emulator = Emulator.load('trained_emulator.hdf5')
    >>> fluxes = emulator.bulk_fluxes
    >>> fluxes = instrumental_broaden(emulator.wl, fluxes, 10)
    >>> eigs = fluxes[:-2]
    >>> flux_mean, flux_std = fluxes[-2:]
    >>> weights, cov = emulator([7054, 4.0324, 0.01])
    >>> X = emulator.eigenspectra * flux_std
    >>> flux = weights @ X + flux_mean
    >>> emu_cov = X.T @ weights @ X

.. note::
    :attr:`Emulator.bulk_fluxes` provides a copy of the underlying arrays, so there is no change to the emulator when bulk processing.


Reference
=========

Emulator
--------

.. autoclass:: Emulator
    :members:
    :special-members: __call__, __str__, __getitem__
