# Starfish

[![Build Status](https://travis-ci.org/iancze/Starfish.svg)](https://travis-ci.org/iancze/Starfish)
[![Doc Status](https://img.shields.io/readthedocs/starfish/latest.svg)](https://starfish.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/iancze/Starfish/badge.svg?branch=master)](https://coveralls.io/github/iancze/Starfish?branch=master)
[![PyPi](https://img.shields.io/pypi/v/astrostarfish.svg)](https://pypi.org/project/astrostarfish/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2221006.svg)](https://doi.org/10.5281/zenodo.2221006)

*Starfish* is a set of tools used for spectroscopic inference. We designed the package to robustly determine stellar parameters using high resolution spectral models.

## Beta Version 0.3

### Documentation

[![Doc Status](https://img.shields.io/readthedocs/starfish/latest.svg?label=latest)](https://starfish.readthedocs.io/en/latest/?badge=latest)
[![Doc Status](https://img.shields.io/readthedocs/starfish/latest.svg?label=develop)](https://starfish.readthedocs.io/en/develop/?badge=develop)

### Citations

If you use this code or derivative components of this code in your research, please cite our [paper](https://ui.adsabs.harvard.edu/abs/2015ApJ...812..128C/abstract) as well as the [code](https://doi.org/10.5281/zenodo.2221006). 

<details>
<summary>BibTex citation</summary>

```
@ARTICLE{2015ApJ...812..128C,
       author = {{Czekala}, Ian and {Andrews}, Sean M. and {Mandel}, Kaisey S. and
         {Hogg}, David W. and {Green}, Gregory M.},
        title = "{Constructing a Flexible Likelihood Function for Spectroscopic Inference}",
      journal = {\apj},
     keywords = {methods: data analysis, methods: statistical, stars: fundamental parameters, stars: late-type, stars: statistics, techniques: spectroscopic, Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = "2015",
        month = "Oct",
       volume = {812},
       number = {2},
          eid = {128},
        pages = {128},
          doi = {10.1088/0004-637X/812/2/128},
archivePrefix = {arXiv},
       eprint = {1412.5177},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2015ApJ...812..128C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@misc{ian_czekala_2018_2221006,
  author       = {Ian Czekala and
                  gully and
                  Kevin Gullikson and
                  Sean Andrews and
                  Jason Neal and
                  Miles Lucas and
                  Kevin Hardegree-Ullman and
                  Meredith Rawls and
                  Edward Betts},
  title        = {{iancze/Starfish: ca. Czekala et al. 2015 release 
                   w/ Zenodo}},
  month        = dec,
  year         = 2018,
  doi          = {10.5281/zenodo.2221006},
  url          = {https://doi.org/10.5281/zenodo.2221006}
}
```

</details>

**Warning!**

There have been major updates since version `0.2`, please see the section of the documentation that regards these changes if you are used to the old version!

### Papers
* [Czekala et al. 2015](https://ui.adsabs.harvard.edu/#abs/2015ApJ...812..128C/abstract)
* [Gully-Santiago et al. 2017](https://ui.adsabs.harvard.edu/#abs/2017ApJ...836..200G/abstract)

Copyright Ian Czekala and collaborators 2013 - 2019 (see [`CONTRIBUTORS.md`](CONTRIBUTORS.md))

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

For the most current release of *Starfish*, use the releases from PyPI

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
virtual environment. We use `pipenv` to manage our environments, to get started clone the repository (and we recommend forking us first)

    $ git clone https://github.com/<your_fork>/Starfish.git starfish
    $ cd starfish

and then create the virtual environment and install pacakges from the `Pipfile` with

    $ pipenv install -d

and to enter the virtual environment, simply issue

    $ pipenv shell

whenever you're in the `starfish` folder.

Take a look through the [issues](https://github.com/iancze/Starfish/issues) if you are looking for a place to start improving *Starfish*!

**Tests**

We use `py.test` for testing; within the virtual environment

    $ pytest


## Contributors

See [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for a full list of contributors.
