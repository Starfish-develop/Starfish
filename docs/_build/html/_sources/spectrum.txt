========
Spectrum
========

.. py:module:: StellarSpectra.spectrum
:synopsis: A package with many different spectra objects for easy manipulation

:mod:`spectrum` is a package containing many different specrum objects. It defines many useful objects
that may be used in the :mod:`grid_tools` package and the :mod:`model` package.


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


Log Lambda spectrum
-------------------

A spectrum spaced linear in log lambda has equal-velocity pixels, meaning that

.. math::

    \frac{v}{c} = \frac{\Delta \lambda}{\lambda}


A log lambda spectrum is defined by the WCS keywords **CDELT1**, **CRVAL1**, and **NAXIS1**. They are related to the
physical wavelengths by the following relationship

.. math::

   \lambda = 10^{{\rm CRVAL1} + {\rm CDELT1} \times i}

where :math:`i` is the pixel index, with :math:`i = 0` referring to the first pixel
and :math:`i = ({\rm NAXIS1} - 1)` referring to the last pixel.


The wavelength array and header keywords are often stored in a ``wl_dict`` dictionary, which looks like
``{"wl":wl, "CRVAL1":CRVAL1, "CDELT1":CDELT1, "NAXIS1":NAXIS1}``.

These keywords are related to various wavelengths by

.. math::

   \frac{v}{c} = \frac{\Delta \lambda}{\lambda} = 10^{\rm CDELT1} - 1

.. math::

   {\rm CDELT1} = \log_{10} \left ( \frac{v}{c} + 1 \right )

.. math::

   {\rm CRVAL1} = \log_{10} ( \lambda_{\rm start})

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