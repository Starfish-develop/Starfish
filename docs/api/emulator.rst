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
    >>> emulator.train()

If you want to perform MLE with a different method, feel free to make use of the general modeling framework provided by the function :meth:`Emulator.get_param_vector`, :meth:`Emulator.set_param_vector`, and :meth:`Emulator.log_likelihood`.

Model spectrum reconstruction
=============================

Once the emulator has been optimized, we can finally use it as a means of interpolating spectra.

.. code-block:: python

    >>> from Starfish.emulator import Emulator
    >>> emulator = Emulator.load('emulator.hdf5')
    >>> flux = emulator.load_flux([7054, 4.0324, 0.01])
    >>> wl = emu.wl


Reference
=========

Emulator
--------

.. autoclass:: Emulator
    :members:
