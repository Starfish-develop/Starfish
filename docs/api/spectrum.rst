========
Spectrum
========

.. py:module:: Starfish.spectrum
   :synopsis: A package to manipulate synthetic spectra

This module contains a few different routines for the manipulation of spectra.

Log lambda spacing
==================

Throughout *Starfish*, we try to utilize log-lambda spaced spectra whenever possible. This is because this sampling preserves the Doppler content of the spectrum at the lowest possible sampling. A spectrum spaced linear in log lambda has equal-velocity pixels, meaning that

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

When resampling wavelength grids that are not log-lambda spaced (e.g., the raw synthetic spectrum from the library) onto a log-lambda grid, the ``dv`` must be calculated. Generally, :meth:`calculate_dv` works by measuring the velocity difference of every pixel and choosing the smallest, that way no spectral information will be lost.

.. autofunction:: Starfish.utils.calculate_dv

.. autofunction:: Starfish.utils.create_log_lam_grid

.. autofunction:: Starfish.utils.calculate_dv_dict


Data Spectrum
=============

The :obj:`Spectrum` holds the data spectrum that you wish to fit. You may read your data into this object in a few ways. First let's introduce the object and then discuss the reading methods.

.. autoclass:: Spectrum
   :members:

First, you can construct an instance using the traditional ``__init__`` method::

    # Read waves, fluxes, and sigmas from your dataset
    # as numpy arrays using your own method.
    waves, fluxes, sigmas = myownmethod()

    myspec = Spectrum(waves, fluxes, sigmas)

Since :meth:`myownmethod` may require a bunch of additional dependencies (e.g, *IRAF*), for convenience you may want to first read your data using your own custom method but then save it to a different format, like ``hdf5``. Since ``HDF5`` files are all the rage these days, you may want to use them to store your entire data set in a single binary file. If you store your spectra in an HDF5 file as ``(norders, npix)`` arrays::

    /
    /waves
    /fluxes
    /sigmas
    /masks

Then can read your data in as::

    myspec = Spectrum.load("myspec.HDF5")

When using HDF5 files, we highly recommended using a GUI program like `HDF View <http://www.hdfgroup.org/products/java/hdfview/index.html>`_ to make it easer to see what's going on.
