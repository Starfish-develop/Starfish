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
This defines the output directory for the many files produced by the sampler during runs of sampling. This is
interpreted as a path and therefore any environment variables will be expanded.

.. code-block:: python

    config.outdir
    'output/'


plotdir
^^^^^^^^^^^^^
This defines the output directory for the plots produced by the different plotting utilities. This is
interpreted as a path and therefore any environment variables will be expanded.

.. code-block:: python

    config.plotdir
    'plots/'


cheb_degree
^^^^^^^^^^^^^

cheb_jump
^^^^^^^^^^^^^

specfmt
^^^^^^^^^^^^^

chunk_ID
^^^^^^^^^^^^^

spectrum_ID
^^^^^^^^^^^^^

instrument_ID
^^^^^^^^^^^^^



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