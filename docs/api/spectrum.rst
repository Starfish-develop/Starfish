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

  \lambda = 10^{{\rm CRVAL1} + {\rm CDELT1} \cdot i}

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

Order
=====

We organize our data into orders which are the building blocks of Echelle spectra. Each order has its own wavelength, flux, optional flux error, and optional mask. 

.. note::
  Typically, you will not be creating orders directly, but rather will be using them as part of a :class:`Spectrum` object.

The way you interact with orders is generally using the properties ``wave``, ``flux``, and ``sigma``, which will automatically apply the order's mask. If you want to reach the underlying arrays, say to create a new mask, use the appropriate ``_``-prepended properties.

.. code-block:: python

  >>> order = Order(...)
  >>> len(order)
  3450
  >>> new_mask = order.mask & (order._wave > 0.9e4) & (order._wave < 4.4e4)
  >>> order.mask = new_mask
  >>> len(order)
  2752

API/Reference
-------------

.. autoclass:: Order
  :members:
  :undoc-members: __len__
  :special-members: __len__

Spectrum
=============

A :obj:`Spectrum` holds the many orders that make up your data. These orders, described by :class:`Order`, are treated as rows in a two-dimensional array. We like to store these spectra in HDF5 files so we recommend creating a pre-processing method that may require any additional dependencies (e.g., *IRAF*) for getting your data into 2-d wavelength arrays calibrated to the same flux units as your spectral library models.

.. code-block:: python

  >>> waves, fluxes, sigmas = process_data("data.fits")
  >>> data = Spectrum(waves, fluxes, sigmas, name="Data")
  >>> data.save("data.hdf5")

Our HDF5 format is simple, with each dataset having shape (norders, npixels):

.. code-block::

  /
    waves
    fluxes
    sigmas
    masks

Whether you save your data to hdf5 or have an external process that saves into the same format above, you can then load the spectrum using

.. code-block:: python

  >>> data = Spectrum.load("data.hdf5")

When using HDF5 files, we highly recommended using a GUI program like `HDF View <http://www.hdfgroup.org/products/java/hdfview/index.html>`_ to make it easer to see what's going on.

To access the data, you can either access the full 2-d data arrays (which will have the appropriate mask applied) or iterate order-by-order

.. code-block:: python

  >>> data = Spectrum(...)
  >>> len(data)
  4
  >>> data.waves.shape
  (4, 2752)
  >>> num_points = 0
  >>> for order in data:
  ...   num_points += len(order)
  >>> num_points == np.prod(data.shape)
  True

API/Reference
-------------

.. autoclass:: Spectrum
  :members:
  :undoc-members: __len__, __getitem__, __setitem__
  :special-members: __len__, __getitem__, __setitem__

