========
Spectrum
========

.. py:module:: Starfish.spectrum
   :synopsis: A package to manipulate synthetic spectra

This module contains a few different routines for the manipulation of spectra.

Log Lambda spectrum
===================

If we consider

A spectrum spaced linear in log lambda has equal-velocity pixels, meaning that

.. math::

  \frac{v}{c} = \frac{\Delta \lambda}{\lambda}

A log lambda spectrum is defined by the WCS keywords **CDELT1**, **CRVAL1**, and **NAXIS1**. They are related to the physical wavelengths by the following relationship

.. math::

  \lambda = 10^{{\rm CRVAL1} + {\rm CDELT1} \times i}

where :math:`i` is the pixel index, with :math:`i = 0` referring to the first pixel and :math:`i = ({\rm NAXIS1} - 1)` referring to the last pixel.

The wavelength array and header keywords are often stored in a ``wl_dict`` dictionary, which looks like ``{"wl":wl, "CRVAL1":CRVAL1, "CDELT1":CDELT1, "NAXIS1":NAXIS1}``.

These keywords are related to various wavelengths by

.. math::

  \frac{v}{c} = \frac{\Delta \lambda}{\lambda} = 10^{\rm CDELT1} - 1

.. math::

  {\rm CDELT1} = \log_{10} \left ( \frac{v}{c} + 1 \right )

.. math::

  {\rm CRVAL1} = \log_{10} ( \lambda_{\rm start})


Many spectral routines utilize a keyword ``dv``, which stands for :math:`\Delta v`, or the velocity difference (measured in km/s) that corresponds to the width of one pixel.

.. math::

  \textrm{dv} = c \frac{\Delta \lambda}{\lambda}

When resampling wavelength grids that are not log-lambda spaced onto a log-lambda grid, the requisite ``dv`` must be calculated. Generally I do it by meausuring the velocity difference of every pixel and choosing the smallest, that way no spectral information will be lost.

.. autofunction:: calculate_dv

.. autofunction:: create_log_lam_grid


Converting user spectra to input format
=======================================

HDF5 files or numpy files.


Full-featured spectrum objects
==============================
These objects may be a little bit slow, but they are designed for convenience to manipulate spectra (convolve, etc)
while being robust to check and handle errors.

.. inheritance-diagram:: BaseSpectrum Base1DSpectrum LogLambdaSpectrum
   :parts: 1

.. autoclass:: BaseSpectrum
   :members:

.. autoclass:: Base1DSpectrum
   :members:


.. py:data:: log_lam_kws

    The parameters that define a log lambda spectrum. This set is use to validate input. By default these are set to

    .. code-block:: python

        log_lam_kws = frozenset(("CDELT1", "CRVAL1", "NAXIS1"))


.. autofunction:: create_log_lam_grid

The :obj:`LogLambdaSpectrum` is more of a workhorse than the previously defined spectra. Because it is regularly spaced
in velocity, it is easier to perform convolution and downsampling operations.

.. autoclass:: LogLambdaSpectrum
   :members:



Fast spectrum objects
=====================
These spectra assume many properties of the data and are built for speedy comparions via :mod:`model`

.. autoclass:: DataSpectrum
   :members:


.. autoclass:: ModelSpectrum
   :members:
