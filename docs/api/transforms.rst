==========
Transforms
==========

These classes and functions are used to manipulate stellar spectra. Users are not expected to directly call these
methods unless they are playing around with spectrums or creating custom methods.

.. py:module:: Starfish.transforms

The Transform Class
===================
Transforms are all subclasses of two classes- :class:`Transform` and :class:`FTransform`

.. autoclass:: Transform
    :members:
    :special-members: __call__


.. autoclass:: FTransform
    :members:
    :special-members: __call__

.. note::
    Make sure you understand the difference in ``__call__`` and ``transform``, especially for :class:`FTransform`.

Wavelength Transformations
==========================
The following transformations are defined by their behavior with the wavelengths.


.. inheritance-diagram:: Transform FTransform Resample DopplerShift
   :parts: 1

.. autoclass:: Truncate
    :show-inheritance:


.. autoclass:: Resample
    :show-inheritance:


.. autoclass:: DopplerShift
    :show-inheritance:


Flux Transformations
====================
The following transformations are defined by their behavior with the fluxes.

.. inheritance-diagram:: Transform FTransform InstrumentalBroaden RotationalBroaden CalibrationCorrect Extinct Scale
   :parts: 1

.. autoclass:: InstrumentalBroaden
    :show-inheritance:

.. autoclass:: RotationalBroaden
    :show-inheritance:

.. autoclass:: CalibrationCorrect
    :show-inheritance:

.. autoclass:: Extinct
    :show-inheritance:

.. autoclass:: Scale
    :show-inheritance:

Helper Methods
==============

Every single class above has a corresponding helper function, e.g. ``doppler_shift`` or
``rotational_broaden``. They act as convenience functions when you just want to apply an action once, such as
while playing around with spectra rather than processing many at once. They all behave the same-

.. code-block:: python

    def transform(wave, flux, *args):
        t = Transform(*args)
        return t(wave, flux)


.. note:: Because the helpers use the ``__call__`` methods, any helpers for :class:`FTransform` subclasses will expect
    normal wavelengths and fluxes in rather than Fourier components.

Sequential Transforms
=====================
The purpose of using classes for transformations is to create elegant sequences of transformations for batch
processing. For example, in :class:`HDF5Creator` we resample the model library fluxes onto a log-lambda grid, then
apply an instrumental broadening kernel, and then resample onto a grid that still retains maximal doppler content but
greatly reduces the number of data points. To facilitate such a transformation, you could string together three
`Transforms`

.. code-block:: python

    from Starfish.grid_tools import PHOENIXGridInterfaceNoAlpha, SPEX
    from Starfish.transforms import Resample, InstrumentalBroaden
    from Starfish.utils import calculate_dv, create_log_lam_grid

    # set up the loglam grid
    grid = PHOENIXGridInterfaceNoAlpha()
    native_dv = calculate_dv(grid.wl)
    loglam = create_log_lam_grid(native_dv, grid.wl.min(), grid.wl.max())['wl']
    resample_loglam = Resample(loglam)

    # set up the instrumental broadening
    inst = SPEX()
    inst_broaden = InstrumentalBroaden(inst)

    # set up the final downsample
    dv_final = inst.FWHM / inst.oversampling
    wave_final = create_log_lam_dict(dv_final, grid.wl.min(), grid.wl.max())['wl']
    resample_final = Resample(wave_final)

    # create our "chained" function
    transform = lambda flux: resample_final(*inst_broaden(*resample_loglam(grid.wl, flux)))

    # apply to the whole library
    transformed_fluxed = map(transform, grid.fluxes)

The NullTransform
-----------------
When creating sequences, you may have a reason to have some "dummy" transform that does not effect the data. That
is exactly to purpose of the :class:`NullTransform`.

.. autoclass:: NullTransform
    :show-inheritance:

.. note:: even though :class:`NullTransform` inherits from :class:`Transform`, it will have the same effect when used
    as a place-holder for an :class:`Ftransform` because it does not touch the data.