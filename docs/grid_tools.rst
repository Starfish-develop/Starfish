==========
Grid Tools
==========

.. py:module:: Starfish.grid_tools
   :synopsis: A package to manipulate synthetic spectra

:mod:`grid_tools` is a module to interface with and manipulate libraries of synthetic spectra.

.. contents::
   :depth: 2

It defines many useful functions and objects that may be used in the modeling package :mod:`model`, such as :obj:`Interpolator`.

Downloading model spectra
=========================

Before you may begin any fitting, you must acquire a synthetic library of model spectra. If you will be fitting spectra of stars, there are many high quality synthetic and empirical spectral libraries available. In our paper, we use the freely available PHOENIX library synthesized by T.O. Husser. The library is available for download here: http://phoenix.astro.physik.uni-goettingen.de/

Because spectral libraries are generally large (> 10 Gb), please make sure you available disk space before beginning the download. Downloads may take a day or longer, so it is recommended to start the download ASAP.

You may store the spectra on disk in whatever directory structure you find convenient, provided you adjust the Starfish routines that read spectra from disk. To use the default settings for the PHOENIX grid, please create a ``libraries`` directory, a ``raw`` directory within ``libraries``, and unpack the spectra in this format::

    libraries/raw/
        PHOENIX/
            WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
            Z+1.0/
            Z-0.0/
            Z-0.0.Alpha=+0.20/
            Z-0.0.Alpha=+0.40/
            Z-0.0.Alpha=+0.60/
            Z-0.0.Alpha=+0.80/
            Z-0.0.Alpha=-0.20/
            Z-0.5/
            Z-0.5.Alpha=+0.20/
            Z-0.5.Alpha=+0.40/
            Z-0.5.Alpha=+0.60/
            Z-0.5.Alpha=+0.80/
            Z-0.5.Alpha=-0.20/
            Z-1.0/


.. _grid-reference-label:

Raw Grid Interfaces
===================

*Grid interfaces* are classes designed to abstract the interaction with the raw synthetic stellar libraries under a common interface. The :obj:`RawGridInterface` class is designed to be extended by the user to provide access to any new grids. Currently there are extensions for three main grids:

 1. `PHOENIX spectra <http://phoenix.astro.physik.uni-goettingen.de/>`_ by T.O. Husser et al 2013
 2. Kurucz spectra by Laird and Morse (available to CfA internal only)
 3. `PHOENIX BT-Settl <http://phoenix.ens-lyon.fr/Grids/BT-Settl/>`_ spectra by France Allard

There are two interfaces provided to the PHOENIX/Husser grid: one that includes alpha enhancement and another which restricts access to 0 alpha enhancement.

.. inheritance-diagram:: RawGridInterface PHOENIXGridInterface KuruczGridInterface BTSettlGridInterface
   :parts: 1

Here and throughout the code, stellar spectra are referenced by a numpy array of parameter values, which corresponds to the parameters listed in the config file.

.. code-block:: python

    my_params = np.array([6000, 3.5, 0.0, 0.0])

Here we introduce the classes and their methods. Below is an example of how you might use the :obj:`PHOENIXGridInterface`.

.. autoclass:: RawGridInterface
   :members:

.. autoclass:: PHOENIXGridInterface
   :members:
   :show-inheritance:

.. autoclass:: PHOENIXGridInterfaceNoAlpha
   :members:
   :show-inheritance:


In order to load a raw file from the PHOENIX grid, one would do

.. code-block:: python

    # if you downloaded the libraries elsewhere, be sure to include base="mydir"
    import Starfish
    from Starfish.grid_tools import PHOENIXGridInterfaceNoAlpha as PHOENIX
    import numpy as np
    mygrid = PHOENIX()
    my_params = np.array([6000, 3.5, 0.0])
    flux, hdr = mygrid.load_flux(my_params)

    In [5]: flux
    Out[5]:
    array([ 4679672.5       ,  4595894.        ,  4203616.5       , ...,
              11033.5625    ,    11301.25585938,    11383.8828125 ], dtype=float32)

    In [6]: hdr
    Out[6]:
    {'PHXDUST': False,
     'PHXLUM': 5.0287e+34,
     'PHXVER': '16.01.00B',
     'PHXREFF': 233350000000.0,
     'PHXEOS': 'ACES',
     'PHXALPHA': 0.0,
     'PHXLOGG': 3.5,
     'PHXTEFF': 6000.0,
     'PHXMASS': 2.5808e+33,
     'PHXXI_N': 1.49,
     'PHXXI_M': 1.49,
     'PHXXI_L': 1.49,
     'PHXMXLEN': 1.48701064748,
     'PHXM_H': 0.0,
     'PHXBUILD': '02/Aug/2010',
     'norm': True,
     'air': True}

    In [7]: mygrid.wl
    Out[7]:
    array([  3000.00133087,   3000.00732938,   3000.01332789, ...,
            53999.27587687,  53999.52580875,  53999.77574063])

.. autoclass:: KuruczGridInterface
   :members:
   :show-inheritance:

.. autoclass:: BTSettlGridInterface
   :members:
   :show-inheritance:


Creating your own interface
---------------------------

The :obj:`RawGridInterface` and subclasses exist solely to interface with the raw files on disk. At minimum, they should each define a :meth:`load_flux` , which takes in a dictionary of parameters and returns a flux array and a dictionary of whatever information may be contained in the file header.

Under the hood, each of these is implemented differently depending on how the synthetic grid is created. In the case of the BTSettl grid, each file in the grid may actually have a flux array that has been sampled at separate wavelengths. Therefore, it is necessary to actually interpolate each spectrum to a new, common grid, since the wavelength axis of each spectrum is not always the same. Depending on your spectral library, you may need to do something similar.


HDF5 creators and Fast interfaces
=================================

While using the :ref:`grid-reference-label` may be useful for ordinary spectral reading, for fast read/write it is best to use HDF5 files to store only the data you need in a hierarchical binary data format. Let's be honest, we don't have all the time in the world to wait around for slow computations that carry around too much data. Before introducing the various ways to compress the spectral library, it might be worthwhile to review the section of the :doc:`spectrum` documentation that discusses how spectra are sampled and resampled in log-linear coordinates.

If we will be fitting a star, there are generally three types of optimizations we can do to the spectral library to speed computation.

1. Use only a range of spectra that span the likely parameter space of your star. For example, if we know we have an F5 star, maybe we will only use spectra that have :math:`5900~\textrm{K} \leq T_\textrm{eff} \leq 6500~\textrm{K}`.
2. Use only the part of the spectrum that overlaps your instrument's wavelength coverage. For example, if the range of our spectrograph is 4000 - 9000 angstroms, it makes sense to discard the UV and IR portions of the synthetic spectrum.
3. Resample the high resolution spectra to a lower resolution more suitably matched to the resolution of your spectrograph. For example, PHOENIX spectra are provided at :math:`R \sim 500,000`, while the TRES spectrograph has a resolution of :math:`R \sim 44,000`.

All of these reductions can be achieved using the :obj:`HDF5Creator` object.

.. autoclass:: HDF5Creator
   :members:

Here is an example using the :obj:`HDF5Creator` to transform the raw spectral library into an HDF5 file with spectra that have the resolution of the *TRES* instrument.

.. code-block:: python

    import Starfish
    from Starfish.grid_tools import PHOENIXGridInterfaceNoAlpha as PHOENIX
    from Starfish.grid_tools import HDF5Creator, TRES


    mygrid = PHOENIX()
    instrument = TRES()

    creator = HDF5Creator(mygrid, Starfish.grid["hdf5_path"], instrument,
    ranges=Starfish.grid["parrange"])

    creator.process_grid()


Once you've made a grid, then you'll want to interface with it via :obj:`HDF5Interface`. The :obj:`HDF5Interface` provides `load_file`  similar to that of the raw grid interfaces. It does not make any assumptions about how what resolution the spectra are stored, other than that the all spectra within the same HDF5 file share the same wavelength grid, which is stored in the HDF5 file as 'wl'. The flux files are stored within the HDF5 file, in a subfile called 'flux'.

.. autoclass:: HDF5Interface
   :members:


Examples
--------

For example, to load a file from our recently-created HDF5 grid

.. code-block:: python

    import Starfish
    from Starfish.grid_tools import HDF5Interface
    import numpy as np
    myHDF5 = HDF5Interface()
    flux = myHDF5.load_flux(np.array([6100, 4.5, 0.0]))

    In [4]: flux
    Out[4]:
    array([ 10249189.,  10543461.,  10742093., ...,   9639472.,   9868226.,
        10169717.], dtype=float32)


Interpolators
=============

The interpolators are used to create spectra in between grid points, for example
``myParams = {"temp":6114, "logg":4.34, "Z": 0.12, "alpha":0.1}``. Note, this currently broken and will remain so unless there is a good use case. For now, it is recommended to use the :doc:`emulator`.

.. autoclass:: Interpolator
   :members:
   :special-members: __call__

For example, if we would like to generate a spectrum with the aforementioned parameters, we would do

.. code-block:: python

    myInterpolator = Interpolator(myHDF5)
    spec = myInterpolator({"temp":6114, "logg":4.34, "Z": 0.12, "alpha":0.1})

Instruments
===========

In order to take the theoretical synthetic stellar spectra and make meaningful comparisons to actual data, we need to convolve and resample the synthetic spectra to match the format of our data. ``Instrument`` s are a convenience object which store the relevant characteristics of a given instrument.

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


Utility Functions
=================

.. autofunction:: chunk_list

.. autofunction:: determine_chunk_log


Wavelength conversions
----------------------

.. autofunction:: vacuum_to_air

.. autofunction:: air_to_vacuum

.. autofunction:: calculate_n


Exceptions
==========

These exceptions will be called if anything is amiss.

.. autoexception:: Starfish.constants.GridError

.. autoexception:: Starfish.constants.InterpolationError
