# Starfish

[![Build Status](https://travis-ci.org/iancze/Starfish.svg)](https://travis-ci.org/iancze/Starfish)

Robustly determine stellar parameters using high resolution spectral models.

Website: http://iancze.github.io/Starfish/

Documentation: http://iancze.github.io/Starfish/current/index.html

Copyright Ian Czekala 2013, 2014

`iczekala@cfa.harvard.edu`

If you wish to use this code in your own research, please email me and I can help you get started. Please bear in mind that this package is under heavy development and features may evolve rapidly. If something doesn't work, please fill an [issue](https://github.com/iancze/Starfish/issues) on the repository. If you would like to contribute to this project (either with bugfixes, documentation, or new features) please feel free to fork the repository and submit a pull request!

**Citation**: if you use this code or derivatives in your research, please cite our paper: Czekala et al. 2014.

# Installation Instructions

## Prerequisites

*Starfish* has several dependencies, however most of them should be satisfied by an up-to-date  
scientific python installation. We highly recommend using the [Anaconda Scientific Python Distribution]
(https://store.continuum.io/cshop/anaconda/) and updating to python 3.3 or greater. This code makes no attempt to work
on the python 2.x series, and I doubt it will if you try. This package has only been tested on Linux,
although it should work fine on Mac OSX and Windows provided you can install the dependencies.

Starfish requires the following Python packages:

* numpy
* scipy
* matplotlib
* h5py
* astropy
* cython
* pyyaml

Unless you actively maintain your own scientific python distribution, I recommend installing the Anaconda
distribution with python 3.3 or greater , obtainable [here](https://store.continuum.io/cshop/anaconda/). All of these
required packages can be installed via Anaconda by doing `conda install pkg` where `pkg` is the name of the package
you want to install.

To make sure you are running the correct version of python, start a python interpreter via the system shell and you
should see something similar

    $ python
    Python 3.4.1 (default, May 19 2014, 17:23:49)
    [GCC 4.9.0 20140507 (prerelease)] on linux  
    Type "help", "copyright", "credits" or "license" for more information.

If your shell says Python 2.x, try using the `python3` command instead of `python`.

## Installation

For now, we recommended building *Starfish* from source on your machine. Soon there will be an option to install via `pip`.

First, if you have not already done so, create a github [user account](https://github.com/) and
[install git](http://git-scm.com/downloads) on your computer.

In order to download a local copy of this repository, ``cd`` to the location where you want it and then do

    git clone git@github.com:iancze/Starfish.git
    cd Starfish

To build the cython extensions

    $ python setup.py build_ext --inplace

Since you may want to edit files in active development, it is best to install in `develop` mode

    $ sudo python setup.py develop

You should now be done. Once the package has stablized, the `develop` command may change to

    $ sudo python setup.py install

To test that you've properly installed *Starfish*, try doing the following inside of a Python interpreter session

    >>> import Starfish

    If you see a blank line, then the package successfully installed. If you see any errors, then something went wrong and please file an [issue](https://github.com/iancze/Starfish/issues).


## Spectral libraries

In order to use *Starfish*, you will need a spectral library with resolution comparable or in excess of that of the spectrograph which acquired your data.

## Features in development**

* Extinction laws
* accretion continuum from veiling
* Moon contamination of spectra
