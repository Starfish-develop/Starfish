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
This defines the smallest iteration amount on the Chebyshev polynomial optimization. See ``Order.update_phi``

.. code-block:: python

    config.cheb_jump
    1e-4

specfmt
^^^^^^^^^^^^^
This is the format string for writing use in writing out information to files during the optimization and sampling.
It must resemble a python format string that will take in two integer parameters- the ``spectrum_id`` and the ``order``.

.. code-block:: python

    config.specfmt
    's{}_o{}'


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



Grid
----


PCA
---


Theta
-----


Theta Jump
----------


Phi
-------


Phi Jump
--------


Comments
--------


Reference
=========

.. autoclass:: Starfish._config.Config
    :members: __init__, copy_file, change_file