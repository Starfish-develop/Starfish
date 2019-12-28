===============
Getting Started
===============

Spectroscopic inference is typically a complicated process, requiring customization based upon the type of spectrum used. Therefore, *Starfish* is not a one-click solution but rather a framework of code that provides the building blocks for any spectroscopic inference code one may write. We provide a few example scripts that show how the *Starfish* code objects may be combined to solve a typical spectroscopic inference problem. This page summarizes the various components available for use and seeks to orient the user. More detailed information is provided at the end of each section.


Citation
========

If *Starfish* or any derivative of it was used for your work, please cite both `the paper <https://ui.adsabs.harvard.edu/abs/2015ApJ...812..128C/abstract>`_ and `the code <https://zenodo.org/record/2221006>`_. We provide a BibTeX formatted file `here <https://github.com/iancze/Starfish/blob/master/CITATION.bib>`_ for your convenience. Thanks!


Installation
============

The source code and installation instructions can be found at the `GitHub repository <https://github.com/iancze/Starfish>`_ for Starfish, but it should be easy enough to run

.. code-block:: console

    pip install astrostarfish

If you prefer to play with some of our new features, check out the code directly from master

.. code-block:: console

    pip install git+https://github.com/iancze/Starfish.git#egg=astrostarfish

or if you prefer an editable version just add the ``-e`` flag to ``pip install``



Obtaining model spectra
========================

Because any stellar synthesis step is currently prohibitively expensive for the purposes of Markov Chain Monte Carlo (MCMC) exploration, *Starfish* relies upon model spectra provided as a synthetic library. However, if you do have a synthesis back-end that is fast enough, please feel free to swap out the synthetic library for your synthetic back-end.

First, you will need to download your synthetic spectral library of choice. What libraries are acceptable are dictated by the spectral range and resolution of your data. In general, it is preferable to start with a raw synthetic library that is sampled at least a factor of ~5 higher than your data. For our paper, we used the freely available `PHOENIX library <http://phoenix.astro.physik.uni-goettingen.de/>`_ synthesized by T. O. Husser. Because the size of spectral libraries is typically measured in gigabytes, I would recommend starting the download process now, and then finish reading the documentation :)

More information about how to download raw spectra and use other synthetic spectra is available in
:doc:`api/grid_tools`. `Starfish` provides a few objects which interface to these spectral libraries.

The Spectral Emulator
=====================

For high signal-to-noise data, we found that any interpolation error can constitute a large fraction of the uncertainty budget (see the appendix of our paper). For lower quality data, it may be possible to live with this interpolation error and use a simpler (and faster) interpolation scheme, such as tri-linear interpolation. However, we found that for sources with :math:`S/N \geq 100` a smoother interpolation scheme was required, and so we developed a spectral emulator.

The spectral emulator works by reconstructing spectra from a linear combination of eigenspectra, where the weight for each eigenspectrum is a function of the model parameters. Therefore, the first step is to deconstruct your spectral library into a set of eigenspectra using principal component analysis (PCA). Thankfully, most of the heavy lifting is already implemented by the `scikit-learn` package.

The next step is training a Gaussian Process to model the reconstruction weights as a function of model parameters
(e.g., effective temperature :math:`T_{\rm eff}`, surface gravity :math:`\log(g)`, and metallicity :math:`[{\rm
Fe}/{\rm H}]`). Because the spectral emulator delivers a probability distribution over the many possible
interpolated spectra, we can propagate interpolation uncertainty into our final parameter estimates. For more on
setting up the emulator, see :doc:`api/emulator`.

Spectrum data formats and runtime
=================================

High resolution spectra are frequently taken with echelle spectrographs, which have many separate spectral orders, or "chunks", of data. This chunking is convenient because the likelihood evaluation of each chunk is independent from the other chunks, meaning that the global likelihood evaluation for the entire spectrum can be parallelized on a computer with many cores.

The runtime of *Starfish* strongly scales with the number of pixels in each chunk. If instead of a chunked dataset, you have a single merged array of more than 3000 pixels, we strongly advise chunking the dataset up to speed computation time. As long as you have as many CPU cores as you do chunks, the evaluation time of *Starfish* is roughly independent of the number of chunks. Therefore, if you have access to a 64 core node of a cluster, *Starfish* can fit an entire ~50 order high-res echelle spectrum in about the same time as it would take to fit a single order. (For testing purposes, it may be wise to use only single order to start, however.)

Astronomical spectra come in a wide variety of formats. Although there is effort to `simplify <http://specutils
.readthedocs.org/en/latest/specutils/index.html>`_ reading these formats, it is beyond the scope of this package to
provide an interface that would suit everyone. *Starfish* requires that the user convert their spectra into one of
two simple formats: *numpy* arrays or HDF5 files. For more about converting spectra to these data formats, see
:doc:`api/spectrum`.

The MCMC driver script
======================

The main purpose of *Starfish* is to provide a framework for robustly deriving model parameters using spectra. The ability to self-consistently downweight model systematics resulting from incorrectly modeled spectral lines is accomplished by using a non-trivial covariance matrix as part of a multi-dimensional Gaussian likelihood function. In principle, one could use traditional non-linear optimization techniques to find the maximum of the posterior probability distribution with respect to the model parameters. However, because one is usually keenly interested in the *uncertainties* on the best-fitting parameters, we must use an optimization technique that explores the full posterior, such as Markov Chain Monte Carlo (MCMC).

Memory usage
============

In our testing, *Starfish* requires a moderate amount of RAM per process (~1 Gb) for a spectrum that has chunk sizes of ~3000 pixels.
