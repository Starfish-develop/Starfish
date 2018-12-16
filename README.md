# Starfish

[![Build Status](https://travis-ci.org/iancze/Starfish.svg)](https://travis-ci.org/iancze/Starfish)
[![Doc Status](https://readthedocs.org/projects/starfish/badge/?version=latest)](https://starfish.readthedocs.io/en/latest/?badge=latest)

*Starfish* is a set of tools used for spectroscopic inference. We designed the package to robustly determine stellar parameters using high resolution spectral models, however there are many potential applications to other types of spectra, such as unresolved stellar clusters or supernovae spectra.

**Beta version 0.2**

Please note that this software package is still under development as more features are being added.

Website: http://iancze.github.io/Starfish/

Documentation: http://iancze.github.io/Starfish/current/index.html

Paper: http://arxiv.org/abs/1412.5177


Copyright Ian Czekala 2013, 2014, 2015, 2016

`iczekala@cfa.harvard.edu`

If you wish to use this code in your own research, please email me and I can help you get started. Please bear in mind that this package is under heavy development and features may evolve rapidly. If something doesn't work, please fill an [issue](https://github.com/iancze/Starfish/issues) on this repository. If you would like to contribute to this project (either with bugfixes, documentation, or new features) please feel free to fork the repository and submit a pull request!

**Citation**: if you use this code or derivative components of this code in your research, please cite our [paper](http://arxiv.org/abs/1412.5177)

# Installation Instructions

## Prerequisites

*Starfish* has several dependencies, however most of them should be satisfied by an up-to-date scientific python installation. We highly recommend using the [Anaconda Scientific Python Distribution](https://store.continuum.io/cshop/anaconda/) and updating to python 3.3 or greater. This code makes no attempt to work on the python 2.x series, and I doubt it will if you try. This package has been tested on Linux and Mac OSX 10.10.

Starfish requires the following Python packages:

* numpy
* scipy
* matplotlib
* h5py
* astropy
* cython
* pyyaml
* scikit-learn
* [emcee](https://github.com/dfm/emcee)
* [corner](https://github.com/dfm/corner.py)

Unless you actively maintain your own scientific python distribution, I recommend installing the Anaconda 
distribution with python 3.6 or greater, obtainable [here](https://store.continuum.io/cshop/anaconda/). All of these required packages can be installed via Anaconda by doing `conda install pkg` where `pkg` is the name of the package you want to install.

To make sure you are running the correct version of python, start a python interpreter via the system shell and you should see something similar

    $ python
    Python 3.6.1 |Anaconda custom (64-bit)| (default, May 11 2017, 13:25:24) [MSC v.1900 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license" for more information.


If your shell says Python 2.x, try using the `python3` command instead of `python`.

## Installation

For now, we recommended building *Starfish* from source on your machine.

First, if you have not already done so install [install git](http://git-scm.com/downloads) on your computer.

In order to download a local copy of this repository, ``cd`` to the location where you want it to live and then do

    git clone https://github.com:iancze/Starfish.git
    cd Starfish

Now we can install the code. We recommend to install using the develop flag so that you can make modifications to 
the source code.
    
    pip install -e .

If you don't care to download the code and would rather install directly into a virtual environment, you can do so with

    pip install git+https://github.com/iancze/Starfiish.git#egg=astrostarfish

or 

    pip install astrostarfish

To test that you've properly installed *Starfish*, try doing the following inside of a Python interpreter session
```python
>>> import Starfish
>>> print(Starfish.__version__)
'0.1'
```
If you see any errors, then something went wrong--please file an [issue](https://github.com/iancze/Starfish/issues).

Now that you've successfully installed the code, please see the [documentation](http://iancze.github.io/Starfish/current/index.html) on how to begin using *Starfish* to solve your spectroscopic inference problem, or head to the [cookbook](http://iancze.github.io/Starfish/current/cookbook.html) for a taste of a typical workflow.


## Contributing
If you are interested in contributing to *Starfish*, first off, thank you! We appreciate your time and effort into 
making our project better. To get set up in a development environment, it is highly recommended to develop in a 
virtual environment. If you are a fan of *pipenv* go ahead and set up using
    
    pipenv install
    
which will automatically process the dependencies from `Pipfile.lock`. If you prefer `conda` go ahead 
with

    conda create --name starfish

if you are on windows:

    activate starfish
   
otherwise

    source activate starfish
    
followed by 
    
    pip install -r requirements.txt
    
Take a look through the [issues](https://github.com/iancze/Starfish/issues) if you are looking for a place to start improving *Starfish*!

## Contributors

See `CONTRIBUTORS.md` for a full list of contributors.