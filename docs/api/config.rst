=============
Configuration
=============

Much of the functionality of this code is based on the ``config.yaml`` file. *Starfish* comes with a default
``config.yaml``. This should NOT be used in your code, and you will receive a warning if you do not provide your own.
To get your own config file, use the provided utility function

.. code-block:: python

    from Starfish import config
    config.copy_file()

This will put a ``config.yaml`` in your current directory for you to edit. For more information about the YAML
standard, visit https://yaml.org/spec/1.2/spec.html

Parameters
==========


General
-------
These are the following parameters derived at the root of the config file

name
^^^^^^^^^^^^^
This is the name of the configuration. It will return a string.

.. code-block:: python

    config.name
    'default'

outdir
^^^^^^^^^^^^^
This defines the output directory for the many files produced by the sampler during runs of sampling.

.. code-block:: python

    config.outdir
    'output/'


plotdir
^^^^^^^^^^^^^
This defines the output directory for the plots produced by the different plotting utilities.

.. code-block:: python

    config.plotdir
    'plots/'

cheb_degree
^^^^^^^^^^^^^
This is the degree of the Chebyshev polynomial to use for flux-calibration correction. See eqn. 4 from Czekala 2015.

.. code-block:: python

    config.cheb_degree
    2

cheb_jump
^^^^^^^^^^^^^
This defines the smallest iteration amount on the Chebyshev polynomial optimization. See :func:`Starfish.parallel.Order.update_phi`

.. code-block:: python

    config.cheb_jump
    1e-4

specfmt
^^^^^^^^
This is the format string for writing use in writing out information to files during the optimization and sampling.
It must resemble a python format string that will take in two integer parameters- the ``spectrum_id`` and the ``order``.

.. code-block:: python

    config.specfmt
    's{}_o{}'

Comments
^^^^^^^^
Any comments for this setup can be stored in this and accessed in the code, otherwise comments can be left
in the YAML file using the ``#`` character.


chunk_ID
^^^^^^^^^^^^^
This is the ID of the chunk that is currently being optimized.

.. code-block:: python

    config.chunk_ID
    0


spectrum_ID
^^^^^^^^^^^^^
This is the ID of the spectrum that is currently being optimized.

.. code-block:: python

    config.spectrum_ID
    0


instrument_ID
^^^^^^^^^^^^^
This is the ID of the Instrument for the data that is currently being optimized. This must represent and index
in the ``config.data.instruments`` list.

.. code-block:: python

    config.instrument_ID
    0


Data
----
This dictionary contains the information about any data files that will be used to fit.

grid_name
^^^^^^^^^
This is the name of the associated grid to fit the data over

.. code-block:: python

    config.data['grid_name']
    'PHOENIX'


files
^^^^^
This is a list of the associated data files

.. code-block:: python

    config.data['files']
    ['data.hdf5']


instruments
^^^^^^^^^^^
These are the instruments used for the data. See `config.instrument_ID` for which instrument to use during fitting. These
should match the name of one of the :class:`Starfish.grid_tools.Instrument` subclasses.

.. code-block:: python

    config.data['instruments']
    ['DCT_DeVeny']


Grid
----
This dictionary contains information about the stellar model grids

raw_path
^^^^^^^^
This is the path to the base directory of the library

.. code-block:: python

    config.grid['raw_path']
    '../libraries/raw/PHOENIX/'


hdf5_path
^^^^^^^^^
This is the path to the hdf5 file to create or created by :class:`Starfish.grid_tools.HDF5Creator`

.. code-block:: python

    config.grid['hdf5_path']
    'grid.hdf5'

parname
^^^^^^^
These are the parameters for the models in the library grid. Examples include effective temperature,
surface gravity, metallicity, and alpha concentration. These can be any strings you like.

.. code-block:: python

    config.grid['parname']
    ['Teff', 'logg', 'Z']


key_name
^^^^^^^^
This is the format string to use when storing parameters in any created hdf5 files with :class:`HDF5Creator`.
It must accept ``len(parname)`` parameters.

.. code-block:: python

    config.grid['key_name']
    'T{0:.0f}_g{1:.1f}_Z{2:.2f}'


parrange
^^^^^^^^
This defines the parameter range for the library. This is meant to represent what subset to use when creating an
:class:`Starfish.grid_tools.HDF5Creator`. This should have shape ``(len(parname), 2)``

.. code-block:: python

    config.grid['parrange']
    [[2300, 3700], [4.0, 5.5], [-0.5, 0.5]]


wl_range
^^^^^^^^
This is the wavelength range to truncate to when creating a file using :class:`Starfish.grid_tools.HDF5Creator`. Note that if this is conflicting
to the wavelength range provided by any :class:`Starfish.grid_tools.Instrument` when processing the grid, the grid will default to the wavelength
range provided by the :class:`Starfish.grid_tools.Instrument`

.. code-block:: python

    config.grid['wl_range']
    [6300, 6360]


buffer
^^^^^^
This signifies how much wavelength past `wl_range` to keep when creating a grid with :class:`Starfish.grid_tools.HDF5Creator`. Note this is
given in Angstrom

.. code-block:: python

    config.grid['buffer']
    50.0


PCA
---
This dictionary stores the parameters for the PCA decomposition and subsequent optimization. See
:class:`Starfish.emulator.PCAGrid` and :class:`Starfish.emulator.Emulator` for more information.


path
^^^^
This is the path to create or to the created PCA file via :func:`Starfish.emulator.PCAGrid`

.. code-block:: python

    config.PCA['path']
    'PCA.hdf5'


threshold
^^^^^^^^^
This is the percentage of variance to be explained by the components of the PCA decomposition. The closer
this value is to ``1.00``, the more eigenspectra will be required in the decomposition. For more information,
see :func:`Starfish.emulator.PCAGrid.create` for more information.

.. code-block:: python

    config.PCA['threshold']
    0.999


priors
^^^^^^
These are the Gamma distribution priors for use in optimizing the spectral emulator. It should have shape
``(len(parname), 2)``. The first value (s) represents the concentration and the second value (r) represents the rate.
The mean value is s / r which represents the length scale of the RBF kernel used to train the gaussian processes
for optimizing the emulator. These should be roughly equal to a few multiples of the
associated parameter's spacing in the grid. For example, the *PHOENIX* grid has an effective temperature spacing
of 100 K below 7000k, so I would want a prior with s / r ~ 300 K. Depending on what values I choose for s and r
given that will define my prior confidence in those estimates. For more information, see https://en.wikipedia.org/wiki/Gamma_distribution.

.. code-block:: python

    config.PCA['priors]
    [[2., 0.0075], [2., 0.75], [2., 0.75]]


Theta
-----
This dictionary defines the initial Theta parameters for the model optimization.

grid
^^^^
These are the initial parameters for the library grid and should have length equal to ``len(parname)``

.. code-block:: python

    config.Theta['grid']
    [2300., 5.0, 0.0]


vz
^^

.. code-block:: python

    config.Theta['vz']
    0.0


vsini
^^^^^

.. code-block:: python

    config.Theta['vsini']
    5.79

logOmega
^^^^^^^^

.. code-block:: python

    config.Theta['logOmega']
    -12.00

Av
^^
This is the initial guess for the parameter defining the extinction applied to the incoming light.

.. code-block:: python

    config.Theta['Av']
    0.0

Theta Jump
----------
This dictionary defines the smallest iterable value for each of the Theta parameters.


grid
^^^^

.. code-block:: python

    config.Theta_jump['grid']
    [3, 0.003, 0.001]


vz
^^

.. code-block:: python

    config.Theta_jump['vz']
    0.01


vsini
^^^^^

.. code-block:: python

    config.Theta_jump['vsini']
    0.01

logOmega
^^^^^^^^

.. code-block:: python

    config.Theta_jump['logOmega']
    1e-4

Av
^^

.. code-block:: python

    config.Theta_jump['Av']
    0.01

Phi
-------
This dictionary defines the initial Phi parameters used for nuisance fitting the global and local covariance kernels.

sigAmp
^^^^^^

.. code-block:: python

    config.Phi['sigAmp']
    1.0

logAmp
^^^^^^
.. code-block:: python

    config.Phi['logAmp']
    -13.6


l
^

.. code-block:: python

    config.Phi['l']
    20.0


Phi Jump
--------
This dictionary defines the smallest iterable value for each of the Phi parameters

sigAmp
^^^^^^

.. code-block:: python

    config.Phi_jump['sigAmp']
    0.025

logAmp
^^^^^^
.. code-block:: python

    config.Phi_jump['logAmp']
    0.01


l
^

.. code-block:: python

    config.Phi_jump['l']
    0.25



Reference
=========

.. autoclass:: Starfish._config.Config
    :members: __init__, copy_file, change_file