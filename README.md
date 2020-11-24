# Starfish
[![Documentation Status](https://readthedocs.org/projects/starfish/badge/?version=latest)](https://starfish.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://github.com/iancze/Starfish/workflows/CI/badge.svg?branch=master)](https://github.com/iancze/Starfish/actions)
[![Coverage Status](https://codecov.io/gh/iancze/Starfish/graph/badge.svg)](https://codecov.io/gh/iancze/Starfish/)
[![PyPI](https://img.shields.io/pypi/v/astrostarfish.svg)](https://pypi.org/project/astrostarfish/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2221005.svg)](https://doi.org/10.5281/zenodo.2221005)

*Starfish* is a set of tools used for spectroscopic inference. We designed the package to robustly determine stellar parameters using high resolution spectral models.

**Warning!**

There have been major, breaking updates since version `0.2.0`, please see [this page](https://starfish.readthedocs.io/en/latest/conversion.html) regarding these changes if you are used to the old version!

### Citations

If you use this code or derivative components of this code in your research, please cite our [paper](https://ui.adsabs.harvard.edu/abs/2015ApJ...812..128C/abstract) as well as the [code](https://doi.org/10.5281/zenodo.2221006). See [`CITATION.bib`](CITATION.bib) for a BibTeX formatted reference of this work.

### Papers
* [Czekala et al. 2015](https://ui.adsabs.harvard.edu/#abs/2015ApJ...812..128C/abstract)
* [Gully-Santiago et al. 2017](https://ui.adsabs.harvard.edu/#abs/2017ApJ...836..200G/abstract)

*If you have used Starfish in your work, please let us know and we can add you to this list!*

Please bear in mind that this package is under heavy development and features may evolve rapidly. If something doesn't work, please fill an [issue](https://github.com/iancze/Starfish/issues) on this repository. If you would like to contribute to this project (either with bugfixes, documentation, or new features) please feel free to fork the repository and submit a pull request!

# Installation Instructions

## Prerequisites

*Starfish* has several dependencies, however most of them should be satisfied by an up-to-date scientific python installation. We highly recommend using the [Anaconda Scientific Python Distribution](https://store.continuum.io/cshop/anaconda/) and updating to 
Python 3.6 or greater. This code makes no attempt to work on the Python 2.x series, and I doubt it will if you try. This package is tested across Linux, Mac OS X, and Windows. 

To make sure you are running the correct version of python, start a python interpreter via the system shell and you should see something similar

    $ python
    Python 3.6.1 |Anaconda custom (64-bit)| (default, May 11 2017, 13:25:24) [MSC v.1900 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license" for more information.
    >>> 

If your shell says Python 2.x, try using the `python3` command instead of `python`.

## Installation

For the most current stable release of *Starfish*, use the releases from PyPI

    $ pip install astrostarfish

If you want to be on the most up-to-date version (or a development version), install from source via

    $ pip install git+https://github.com/iancze/Starfish.git#egg=astrostarfish


To test that you've properly installed *Starfish*, try doing the following inside of a Python interpreter session

```python
>>> import Starfish
>>> Starfish.__version__
'0.3.0'
```

If you see any errors, then something went wrong--please file an [issue](https://github.com/iancze/Starfish/issues).

Now that you've successfully installed the code, please see the [documentation](https://starfish.readthedocs.io/en/latest/) on how to begin using *Starfish* to solve your spectroscopic inference problem.

# Contributing
If you are interested in contributing to *Starfish*, first off, thank you! We appreciate your time and effort into
making our project better. To get set up in a development environment, it is highly recommended to develop in a
virtual environment. We use `pipenv` (pending a better PEP 517/518 compliant tool) to manage our environments, to get started clone the repository (and we recommend forking us first)

    $ git clone https://github.com/<your_fork>/Starfish.git starfish
    $ cd starfish

and then create the virtual environment and install all the packages and developer dependencies from the `Pipfile` with

    $ pipenv install -d

and to enter the virtual environment, simply issue

    $ pipenv shell

whenever you're in the `starfish` folder.

We also enforce the `black` [code style](https://github.com/python/black). This tools allows automatically formatting everything for you, which is much easier than caring about it yourself! We have a [pre-commit](https://pre-commit.com/) hook that will *blacken* your code before you commit so you can avoid failing the CI tests because you forgot to format. To use this, just install the hook with 

    $ pipenv run pre-commit install

From then on, any commits will format your code before succeeding!

Take a look through the [issues](https://github.com/iancze/Starfish/issues) if you are looking for a place to start improving *Starfish*!

**Tests**

We use `pytest` for testing; within the virtual environment

    $ pytest

Note that we use the `black` code style and our CI testing will check that everything is formatted correctly. To check your code

    $ pytest --black

although if you follow the instructions for using *pre-commit* you should have no issues.


## Contributors

See [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for a full list of contributors.
