==========
Grid Tools
==========

.. py:module:: grid_tools
   :synopsis: A package to manipulate synthetic stellar spectra

:mod:`grid_tools` is a package to manipulate synthetic stellar spectra. It defines many useful functions and objects
that may be used in the modeling package :mod:`model`, such as :obj:`Interpolator`.

Module level methods
====================

These methods are meant to be used for low-level access to spectral libraries. Generally there is no error checking.

.. automodule:: grid_tools
   :members: resample_and_convolve, load_BTSettl, load_flux_full, chunk_list

.. autofunction:: gauss_taper

.. autofunction:: idl_float

Wavelength conversion methods
-----------------------------

.. autofunction:: vacuum_to_air

.. autofunction:: air_to_vacuum

.. autofunction:: calculate_n


Grid Interfaces
===============

Overview
--------

The *grid interfaces* are designed to abstract the interaction with raw synthetic stellar libraries that one might
download. The :obj:`RawGridInterface` is designed to be extended by the user to provide access to any new grids.
Currently there are extensions for three main grids:

 1. `PHOENIX spectra <http://phoenix.astro.physik.uni-goettingen.de/>`_ by T.O. Husser et al 2013
 2. Kurucz spectra by Laird and Morse
 3. `PHOENIX BT-Settl <http://phoenix.ens-lyon.fr/Grids/BT-Settl/>`_ spectra by France Allard

.. inheritance-diagram:: RawGridInterface PHOENIXGridInterface KuruczGridInterface BTSettlGridInterface
   :parts: 1

Stellar spectra are referenced by a dictionary of parameter values. The use of the *alpha* parameter (denoting alpha
enhancement :math:`[{\rm \alpha}/{\rm Fe}]`) is optional and defaults to `0.0`.

    .. code-block:: python

        my_params = {"temp":6000, "logg":3.5, "Z":0.0, "alpha":0.0} #or
        other_params = {"temp":4000, "logg":4.5, "Z":-0.2}

.. py:data:: grid_parameters

    The variables that define a grid. This set is use to validate input. By default these are set to

    .. code-block:: python

        grid_parameters = frozenset(("temp", "logg", "Z", "alpha"))

.. py:data:: pp_parameters

    The variables that define a grid. This set is use to validate input. By default these are set to

    .. code-block:: python

        pp_parameters = frozenset(("vsini", "FWHM", "vz", "Av", "Omega"))

.. py:data:: all_parameters

    The variables that define a grid. This set is use to validate input. By default these are set to

    .. code-block:: python

        all_parameters = grid_parameters | pp_parameters #union

.. _raw-grid-interfaces-label:

Raw Grid Interfaces
-------------------

.. autoclass:: RawGridInterface
   :members:

.. autoclass:: PHOENIXGridInterface
   :members:
   :show-inheritance:

In order to load a file from the PHOENIX grid, one would do

.. code-block:: python

    myPHONEIXgrid = PHOENIXGridInterface(air=True, norm=True)
    spec = myPHOENIXgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0, "alpha":0.0}) #Base1DSpectrum object



HDF5 creator and reader
=======================

While using the :ref:`raw-grid-interfaces-label` may be useful for ordinary spectral reading, for fast read/write it
is best to use HDF5 files, which store data in a hierarchical data format.

The :obj:`HDF5GridCreator` uses a raw grid interface to load all of the available files, processes these files to a
common format and wavelength sampling, and saves them as individual datasets in the HDF5 file with all available metadata.

The :obj:`HDF5Interface` provides `load_file` method similar to that of the raw grid interfaces.

.. autoclass:: HDF5GridCreator
   :members:

For example, to create a master grid for the PHOENIX spectra, we use our previously created :obj:`PHOENIXGridInterface`
and create a new :obj:`HDFGridCreator`. Then we run ``process_grid()`` to process all of the raw files on disk into an
HDF5 file.

.. code-block:: python

    #First, load a test file to determine the wldict
    spec = myPHOENIXgrid.load_file({"temp":5000, "logg":3.5, "Z":0.0,"alpha":0.0})
    wldict = spec.calculate_log_lam_grid() #explaned in the model.py documentation


    HDF5Creator = HDF5GridCreator(myPHOENIXgrid, filename="test.hdf5", wldict=wldict, nprocesses=10, chunksize=1)
    HDF5Creator.process_grid()


Once an HDF5 file has been created using an :obj:`HDF5GridCreator`, spectra can be easily accesed from it through an
:obj:`HDF5Interface` instance.

.. autoclass:: HDF5Interface
   :members:

For example, to load a file from our HDF5 grid

.. code-block:: python

    myHDF5 = HDF5Interface("test.hdf5")
    spec = myHDF5.load_file({"temp":6100, "logg":4.5, "Z": 0.0, "alpha":0.0})


Interpolators
=============
The interpolators are used to create spectra in between grid points, for example
``myParams = {"temp":6114, "logg":4.34, "Z": 0.12, "alpha":0.1}``.


.. autoclass:: IndexInterpolator
   :members:
   :special-members: __call__

.. autoclass:: Interpolator
   :members:
   :special-members: __call__

For example, if we would like to generate a spectrum with the aforementioned parameters, we would do

.. code-block:: python

    myInterpolator = Interpolator(myHDF5)
    spec = myInterpolator({"temp":6114, "logg":4.34, "Z": 0.12, "alpha":0.1})

Instruments
===========

In order to take the theoretical synthetic stellar spectra and make meaningful comparisons to actual data, we need to
convolve and resample the synthetic spectra to match the format of our data. *Instruments* are a convenience object
which store the relevant characteristics of a given instrument. These objects will be later passed to methods which
convolve raw spectra by the instrumental profile.

.. inheritance-diagram:: Instrument KPNO TRES Reticon
   :parts: 1

.. autoclass:: Instrument
   :members:
   :special-members: __str__

.. autoclass:: TRES
   :members:
   :show-inheritance:

.. autoclass:: KPNO
   :members:
   :show-inheritance:

.. autoclass:: Reticon
   :members:
   :show-inheritance:


FITS creator
------------

Some techniques using synthetic spectra require them as input in FITS files. The :obj:`MasterToFITSProcessor` uses a
:obj:`HDF5Interface` in order interface to a master stellar grid stored in HDF5 format. Given a parameter set, the object
will create a FITS file storing a spectrum.

.. autoclass:: MasterToFITSProcessor
   :members:

For example, to process all of the PHOENIX spectra into FITS files suitable for the :obj:`KPNO` instrument, we would do

.. code-block:: python

    myInstrument = KPNO()

    mycreator = MasterToFITSProcessor(interpolator=myInterpolator, instrument=myInstrument,
    outdir="outFITS/", points={"temp":np.arange(2500, 12000, 250), "logg":np.arange(0.0, 6.1, 0.5),
    "Z":np.arange(-1., 1.1, 0.5), "vsini": np.arange(0.0, 16.1, 1.)}, processes=32)

    mycreator.process_all()


Exceptions
==========

.. autoexception:: GridError

.. autoexception:: InterpolationError
