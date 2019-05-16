#####
Setup
#####

This guide will show you how to get up and running with the grid tools and interfaces provided by *Starfish*. 

Getting the Grid
================

To begin, we need a spectral model library that we will use for our fitting. One common example are the PHOENIX models, most recently computed by T.O. Husser. We provide many interfaces directly with different libraries, which can be viewed in :ref:`grid-reference-label`.

As a convenience, we provide a helper to download PHOENIX models from the Goettingen servers

.. code-block:: python

    import itertools
    from Starfish.grid_tools import download_PHOENIX_models

    T = [5000, 5100, 5200]
    logg = [4.0, 4.5, 5.0]
    Z = [0]
    params = itertools.product(T, logg, Z)
    download_PHOENIX_models(path='PHOENIX', parameters=params)

We now want to set up a grid interface to work with these downloaded files!

.. code-block:: python

    from Starfish.grid_tools import PHOENIXGridInterfaceNoAlpha

    grid = PHOENIXGridInterfaceNoAlpha(path='PHOENIX')

From here, we will want to set up our HDF5 interface that will allow us to go on to using the spectral emulator, but first we need to determine our model subset and instrument.

Setting up the HDF5 interface
=============================

We set up an HDF5 interface in order to allow much quicker reading and writing than compared to loading FITS files over and over again. In addition, when considering the application to our likelihood methods, we know that for a given dataset, any effects characteristic of the instrument can be pre-applied to our models, saving on computation time during the maximum likelihood estimation. 

Looking towards our fitting examples, we know we will try fitting some data from the `TRES spectrograph <http://tdc-www.harvard.edu/instruments/tres/>`_. We provide many popular spectrographs in our grid tools, including TRES.


Let's also say that, for a given dataset (in our future examples we use WASP 14 so let's consider that), we want to only use a reasonable subset of our original model grid. WASP 14 is currently labeled as an F5V star, so let's create a subset around that classification.

.. code-block:: python
    
    from Starfish.grid_tools.instruments import TRES
    
    # Parameters are Teff, logg, and Z
    ranges = [
        [5900, 7400],
        [3.0, 6.0],
        [-1.0, 1.0]
    ]
    inst = TRES()

Now we can create and process our HDF5Interface

.. code-block:: python

    from Starfish.grid_tools import HDF5Creator

    creator = HDF5Creator(grid, 'F_TRES_grid.hdf5', instrument=inst, ranges=ranges)
    creator.process_grid()


Setting up the Spectral emulator
================================

Once we have our pre-processed grid, we can make our spectral emulator and train its GP hyperparameters.

.. code-block:: python

    from Starfish.grid_tools import HDF5Interface
    from Starfish.emulator import Emulator

    grid = HDF5Interface('F_TRES_grid.hdf5')
    emu = Emulator.from_grid(grid)
    emu.train()
    emu.save('F_TRES_emu.hdf5')

.. warning::

    Training the emulator will take on the order of minutes to complete. The more eigenspectra that are used as well as the resolution of the spectrograph will mainly dominate this runtime. 


Once we have our trained emulator, we can move on to the modeling steps for our data.
