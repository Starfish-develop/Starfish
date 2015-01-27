=================
Spectral Emulator
=================

.. py:module:: Starfish.emulator
   :synopsis: Return a probability distribution over possible interpolated spectra.

The spectral emulator can be likened to the engine behind *Starfish*. While the novelty of *Starfish* comes from using Gaussian processes to model and account for the covariances of spectral fits, we still need a way to produce model spectra by interpolating from our synthetic library. While we could interpolate spectra from the synthetic library using something like linear interpolation in each of the library parameters (:math:`T_{\rm eff}`, :math:`\log(g)`, :math:`[{\rm Fe}/{\rm H}]`), it turns out that high signal-to-noise data requires something more sophisticated. This is because the error in any interpolation can constitute a significant portion of the error budget. This means that there is a chance that non-interpolated spectra (e.g., the parameters of the synthetic spectra in the library) might be given preference over any other interpolated spectra, and the posteriors will be peaked at the grid point locations. Because the spectral emulator returns a probability distribution over possible interpolated spectra, this interpolation error can be quantified and propagated forward into the likelihood calculation.

Eigenspectra decomposition
==========================

The first step of configuring the spectral emulator is to choose a subregion of the spectral library corresponding to the star that you will fit. Then, we want to decompose the information content in this subset of the spectral library into several *eigenspectra*. [Figure A.1 here].

The eigenspectra decomposition is performed via Principal Component Analysis (PCA). Thankfully, most of the heavy lifting is already implemented by the ``scipy`` package.

:obj:`PCAGrid` implements the functionality to create the eigenspectra grid from a synthetic library, and then later query eigenspectra from it.

.. autoclass:: PCAGrid
   :members:

Model spectrum reconstruction
=============================

.. autoclass:: WeightEmulator
   :members:

.. autoclass:: Emulator
   :members:
