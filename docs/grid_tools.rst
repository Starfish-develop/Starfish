==========
Grid Tools
==========

.. py:module:: Starfish.grid_tools
   :synopsis: A package to manipulate synthetic stellar spectra

:mod: `grid_tools` is a package to manipulate synthetic stellar spectra. It defines many useful functions and objects
 that may be used in the modeling package :mod:`model`, such as :obj:`Interpolator`.

Downloading model spectra
=========================

PHOENIX library.

Module level methods
====================

These methods are meant to be used for low-level access to spectral libraries.
Generally there is only a small amount of error checking.

.. automodule:: Starfish.grid_tools
   :members: resample_and_convolve, load_BTSettl, load_flux_full, chunk_list

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

All interactions with the grid are designed to be abstracted such that the user can easily extend to add a new
spectral grid simply by writing a new class which inherits :obj:`RawGridInterface`.

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

.. autoclass:: KuruczGridInterface
   :members:
   :show-inheritance:

.. autoclass:: BTSettlGridInterface
   :members:
   :show-inheritance:

In order to load a raw file from the PHOENIX grid, one would do

.. code-block:: python

    myPHONEIXgrid = PHOENIXGridInterface(air=True, norm=True)
    spec = myPHOENIXgrid.load_file({"temp":6000, "logg":3.5, "Z":0.0, "alpha":0.0}) #Base1DSpectrum object


The :obj:`RawGridInterface` and subclasses exist solely to interface with the raw files on disk. At minimum,
they should each define a :method:`load_flux` method, which takes in a dictionary of parameters and returns a
flux array and a dictionary of whatever information may be contained in the file header.

Under the hood, each of these is implemented differently depending on how the synthetic grid is created. In the case
of the BTSettl grid, each file in the grid may actually have a flux array that has been sampled at separate
wavelengths. Therefore, it is necessary to actually interpolate each spectrum to a new, common grid, since
the wavelength axis of each spectrum is not always the same.


HDF5 creators and interfaces
============================

While using the :ref:`raw-grid-interfaces-label` may be useful for ordinary spectral reading, for fast read/write it
is best to use HDF5 files, which store data in a hierarchical data format.

The :obj:`HDF5GridStuffer` uses a raw grid interface to load all of the available files and saves them as individual
datasets in the HDF5 file with all available metadata. If the wavelength format is different between individual
files in the raw grid (BTSettl), it does minimal processing to unify them to the same wavelength grid. It aims to
simply reproduce relevant sections of the raw grid (say, full optical, or certain temperature and metallicity ranges)
in a unified file format (HDF5), so that it can be easily stored and transferred from the location of the files
(say,  some external disk) to a laptop.

.. autoclass:: HDF5GridStuffer
   :members:

The :obj:`HDF5Interface` provides `load_file` method similar to that of the raw grid interfaces. It does not make any
assumptions about how what resolution the spectra are stored, other than that the all spectra within the same HDF5 file
share the same wavelength grid, which is part of the HDF5 file in 'wl'. The flux files are stored within the HDF5
file, in a subfile called 'flux', and are labelled as a dataset with the format

    t{temp:.0f}g{logg:.1f}z{Z:.1f}a{alpha:.1f}

.. autoclass:: HDF5Interface
   :members:


The :obj:`HDF5InstGridCreator` is designed to use the :obj:`HDFInterface` to interface to an HDF5 file which has
already been created by :obj:`HDF5GridStuffer`, and then process the relevant flux files to a sub grid that for the
instrument under consideration.

.. autoclass:: HDF5InstGridCreator
   :members:

Once the grid subsampled for a specific instrument has been created, then it too can be read by the same
:obj:`HDF5Interface` class.


Examples
--------

For example, to create a master grid for the PHOENIX spectra, we use our previously created :obj:`PHOENIXGridInterface`
and create a new :obj:`HDFGridStuffer`. Then we run ``process_grid()`` to process all of the raw files on disk into an
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


The :obj:`HDF5GridCreatorDep` is designed to transfer the raw grid to a sub sampled grid. It is deprecated,
but was used for the :obj:`MasterToFITSGridCreator` methods and so it remains.

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

   .. attribute:: self.wl_dict

    A wl_dict that fits the instrumental properties with the correct oversampling.

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
============

Some techniques using synthetic spectra require them as input in FITS files. The :obj:`MasterToFITSProcessor` uses a
:obj:`HDF5Interface` in order interface to a master stellar grid stored in HDF5 format. Given a parameter set, the object
will create a FITS file storing a spectrum.

.. autoclass:: MasterToFITSIndividual
   :members:


To process an individual spectrum to a FITS file

.. code-block:: python

    myInstrument = KPNO()
    myInterpolator = Interpolator(myHDF5Interface)
    KPNOcreator = MasterToFITSIndividual(interpolator=myInterpolator, instrument=myInstrument)

    params = {"temp":6000, "logg":4.5, "Z":0.0, "vsini":8}
    KPNOcreator.process_spectrum(params, out_unit="f_nu_log")


.. autoclass:: MasterToFITSGridProcessor
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

.. autoexception:: Starfish.constants.GridError

.. autoexception:: Starfish.constants.InterpolationError
