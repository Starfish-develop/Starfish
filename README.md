StellarSpectra
==============

Repository for techniques to fit stellar spectra using a likelihood function taking into account a non-trivial covariance matrix.

Copyright Ian Czekala 2013, 2014

`iczekala@cfa.harvard.edu`
github id: iancze

**Documentation** for this package is available at ReadTheDocs.

# Using git to interact with this repository

First, if you have not already done so, create a github [user account](https://github.com/) and [install git](http://git-scm.com/downloads) on your computer.

If you are at the CfA and using the CF, you can use the version of git installed by Tom Aldcroft by adding the following to your ``.cshrc``

    set path=(/data/astropy/ska/arch/x86_64-linux_CentOS-5/bin $path)

If this is the first time you are using github on your computer, you will also want to set up your [SSH keys](https://help.github.com/articles/generating-ssh-keys).

## Getting the files

In order to download a local copy of this repository, so that you may edit or add files, ``cd`` to the location where you want it and then do

    git clone git@github.com:iancze/StellarSpectra.git
    cd StellarSpectra

and you should see a copy of everything that is on github.com.

## Editing the files

You can edit files however you might normally interact with files on your computer, even creating new files and deleting other ones (though you should first have a good reason to do this).

When you have finished your edits, you will want to ``add`` and then ``commit`` your changes to your local copy of the repository.

If you have created any new files, first do

    git add my_new_file

Then to commit your changes (including any changes to previously existing files) do

    git commit -am "my commit message here"

It is helpful to write a brief but descriptive commit message, such as, "added Figure 2 caption" or something of the like.

To ``push`` these changes to the remote copy of the repository on github, you then do

    git push origin master

If everything worked, you should be able to go to the repository on github and view your changes. If something went wrong, see the next section for help.

## Updating your files

The previous instructions for editing files will normally work fine as long as there haven't been any changes to the remote github repository while you were editing the local copy. In single user-mode, this is generally the case and everything should work. For now, I'll try to restrict my own edits to a separate branch so that no conflicts occur. In the future, however, this will likely not be the case and we will want to utilize the full power of git and github.

Because the documentation can explain this better than I can, I would recommend reading

1. [Fetching and merging](https://help.github.com/articles/fetching-a-remote)
2. [Dealing with non-fast forward errors](https://help.github.com/articles/dealing-with-non-fast-forward-errors)
3. [Pushing to a remote](https://help.github.com/articles/pushing-to-a-remote)

Generally, what these documents boil down to is that if some time has elapsed since you last did a ``git pull``, and another collaborator has made updates to the repository on github, then you must first update your own repository with that collaborators edits. Before doing ``git push``, you can update your local copy by ``cd``ing to the StellarSpectra directory and executing

    git pull origin master

which will ``pull`` the latest files from the ``origin`` branch (a fancy name for the version that is on github) and automatically merge them into your local files. If there are conflicts with the code you have edited, you must manually merge these files (see #1 and #2) and then do another

    git commit -am "merged from upstream"

And then finally

    git push origin master


# Information for compiling and editing the paper

## Compiling

Compiling the paper with ``latex`` should be as easy as ``cd``ing to the ``tex`` directory and then doing ``make``, as long as you have the necessary ``latex`` packages to compile a typical AASTEX file. The Makefile is inspired by those in authored by @davidwhogg.

### Compiling on the CF

However, if you use the outdated (2007) tex distribution that is default to a CF computer, the document will likely not compile. Therefore, if you are compiling this on the CF, then you might want to do the following:

From the CF:
As of August 2012, the CF has an installation of TeX Live 2012 (version 20120701) available in ``/opt/texlive/2012``. Symlinks to executables for this TeX system have been placed in ``/opt/bin`` on all CF managed Solaris and Linux computers.

This version can be added to your shell search path by adding ``/opt/bin`` first in your ``$path`` and running the ``rehash`` command:

    set path = ( /opt/bin $path )
    rehash

It is not recommended that you add this to your ``.cshrc``, so I guess you will need to run this each time you open up a shell.

## Editing the paper

Proposed Style guide: since the ApJ now accepts PDF format figures, I propose to eschew the horrible EPS format entirely and just use PDF figures!

Since git works by doing a line-by-line ``diff`` between files, it is best to use hard word wraps of size 80 characters to minimize clutter and make it easier to compare changes. In addition, each sentence should start on its own line. If the sentence is longer than 80 characters, then the subsequent lines of the sentence should be indented by one space.

# Summary of Approach

We estimate stellar parameters by taking a Bayesian approach to generate a foreward model of the data (and systematic errors) and sample this posterior using the `emcee` implementation of the Markov Chain Monte Carlo ensemble sampler by Goodman and Weare (2010).

# Dependencies

Currently, StellarSpectra has many dependencies, however most of them should be satisfied by an up-to-date scientific python installation. We highly recommend using the [Anaconda Scientific Python Distribution](https://store.continuum.io/cshop/anaconda/) and updating to python 3.3. This code makes no effort to work on the python 2.x series, and I doubt it will if you try. This package has only been tested on Linux, although it should be able to work on Mac OSX and Windows provided you can install the dependencies.

StellarSpectra requires the following Python packages:

* numpy†
* scipy†
* jinja2†
* matplotlib†
* h5py†
* astropy†
* cython†
* pyyaml†
* emcee


Those marked with a † are included in the latest Anaconda distribution. The others should be available from ``pip`` or from the project websites.

## SuiteSparse

StellarSpectra also requires the [SuiteSparse](https://www.cise.ufl.edu/research/sparse/SuiteSparse/) sparse matrix library, which is written in C. If you have administrator priveledges, installing SuiteSparse from a package manager (apt-get, yum, pacman, etc...) is likeley the easiest. Otherwise, you can download the SuiteSparse package and instal from source to a user defined directory. If you are on the CfA CF network, you may be able to use the libraries I have linked to below.

If you are installing SuiteSparse on a computer for which you do not have root access, then you can change the install directory by viewing the SuiteSparse README.txt, and editing `SuiteSparse_config.mk` before building.

# Installation

## Normal install

Follow these if you have installed the Python packages above and were able to install SuiteSparse to a normal location using your package manager (`/usr/local/include` or something similar for Mac OS X).

To build

    $ python setup.py build_ext --inplace

For now, since we are in active development, it is best to install in `develop` mode

    $ sudo python setup.py develop

You should now be done. Once the package has stablized, the `develop` command may change to

    $ sudo python setup.py install


## Modified install

If you needed to install SuiteSparse to a different location than the default, then you can install with the following commands, depending on where you installed the library.

For example, if you installed SuiteSparse to mylib, then you would call setup.py as

    python setup.py build_ext --inplace -Lmydir/lib -Imydir/include

If you want to try using my SuiteSparse directories on the CfA CF network,

    python setup.py build_ext --inplace -L/pool/scout0/.build/lib -I/pool/scout0/.build/include

Then, the install command is

    $ python setup.py develop --user


# Tasks and development

# Before the 1.0 paper release

* auto function in iPython notebook to display figure inline and save it to svg, pdf, and png formats.

# Running many jobs in parallel

* Extend HDF5cat to chebyshev and global cov
* Later extend to sorting regions?

* how to find autocorrelation time of a single chain?
* do Gelman-Rubin statistic of many chains?


# List of tests

From the testing, it seems like using the Matern kernel really doesn't do much. In the case of the PHOENIX spectrum,
it frees up some extra space, but it doesn't really free you from *bias*, which is what is important. Instead,
it is the line identifications and downweighting that should be used.

Work on perfecting regions

Interestingly, Kurucz with the covariant structure and without the covariant structure look exactly the same. Is this
 because the noise is so low? What do the residuals look like?  LogAmp = -14.56. Actual residuals are at what scale?

Tests are interesting. In some sense, the Matern kernel really doesn't matter all that much. Its allows the error
bars to inflate. In many cases, the identifying and tracking bad lines is the more important behavior.


Does this same behavior happen with the PHOENIX spectra? Running tests to find out.


# Gl51

* logg fixed to 5.0, low polynomial

* to do a no global, save `base_lnprob` as `no_global`

* can we tune the cadences somehow?

## working test with IRTF spex M dwarf

* What regions of the spectrum did Rojas-Ayala actually use? We should be limiting ourselves to just these regions.

This will tell us what the residuals actually look like, which is important for developing a kernel to track them.

In this case, I think it's worthwhile to use a Gaussian-tapered Matern kernel, since that looks more ragged.

# Priors on the parameters

* Chebyshev's in particular

## Regions

The regions need priors on sigma being < 15 km/s, and keeping mu to within something of 1AA or some w/ in velocity
space.

Where do we put these priors? Can each region have a value, log_prior, that it carries around with it,
and is summed during the CovarianceMatrix.evaluate() step?

order 24 is just ridiculous. When you mask these regions you get a very low answer (Z ~ 0.93) and logg ~ 2.7 and temp
 ~ 5800. You can't be sure if it's burned in or not, but basically this isn't a good order to test things on. I think
  we'll have to try this with a different order, perhaps order 23.

* Let's try to get a well-converged spectrum for WASP-14, order 23, PHOENIX
    ie, all 12 bad "regions" are properly instantiated
* keep an anchor on mu that makes sense relative to the PSF, +/- 3 ang for TRES is way too big. Need something for
    SPEX.

* use flot annotations to mark a vertical line on the graph

    * needs to pan + zoom with the script
    * also plot a vertical range that is 3X the global covariance height, or whatever our cutoff is (maybe this is
    the first thing to do)

* This also begets a need to have a proper visualization tool for a single order. This is Jekyll, bootstrap, etc.

Flot works well enough, and annotations are easy. Use the "markings" and the "fillBetween" plugins to get this done.

Takes as input flatchains, model.json, input. Goes through all region areas, and maybe at first glance just plots
 vertical lines on the plot.

I feel like this should be a separate plotting routine. Take in `*.yaml` file, input script,
load all residuals and plot the regions on top, along with error envelopes.

simple Gaussian envelope (high + low) superimposed where line is? (Filled areas for lines)

add logic to be destroyed when current value of amplitude goes below the global kernel

Good figure for paper will be a figure similar to fig 3 (correlations panel) for the regions. It will show the
envelope (random draws) of the spectral line kernel around the strong residual.

Masks do not play well when instantiating regions. It's probably because of a length mis-match.


## get spectral libraries (BTSettl, PHOENIX, Kurucz) in the right form

* use Julia to do the spline interpolation to a finer spaced grid, at high resolution.


#Alternate sampling stratagies

* In theory, if this step is slow, we could sample all of the parameters, for all of the regions (and all of the
 global parameters) at the same time (perhaps using a HMC algorithm). That way we get around the largest time cost of
  the update_cov step, which for ~17 orders becomes a sizeable amount of time.

* alternatively, for each region, it might be possible to actually isolate the exact chunk that we are sub-sampling.
Although I have a feeling this won't be all that much faster anyway.

After converging the global and stellar parameters, we could switch entirely to sampling the hyperparameters. Or
sample them less frequently. I suppose the point is simply to track how they evolve.
This is why reading in a couple different model.json files to visualize the current state of things might be helpful.


## text in paper

* redo the FFT/convolve section to include references to FFT and overlap-add, overlap-save. Note that these are actually corrections to the convolved spectra.

* text for section 3, testing

* text for section 4, discussion


## Desireable plotting tools and output analysis, after 1.0 release

Using `flot`,

* Output, plot the covariance matrix as a 1D slice through row=constant (how correlated is a given pixel with everything else?)
* Scroll through rows?

* create my own "linelist" using an astropy table, easily plot line regions of data in a "gallery"
* change masking list into an astropy table
* fit Telluric sky spectrum, with no wavelength shift

* an animated diagram of age tracks in the HR diagram using D3? For talks and webpage

* add visualization of non-stationary kernels to gp.js

# Things to check

* Flux interpolation errors (a systematic spectrum) (see notes + figure from 9/9/13)
* distribution of interpolation errors (histogram)
* sort out usage of C.kms_air and C.kms_vac

# Covariance kernels and line logic

* convert covariance function to work on delta velocity instead of wavelength difference

* convert kernel to run without `h`, or `h` fixed to zero. Maybe `george` has a smart way of encoding kernels with an arbitrary amount of parameters.

* Is it a problem when the sigma's overlap? Can we do the tapering better?

Cleanup logic: make some assertion that the amplitude of the region can't be less than the global covariance height? If it ever makes it to this part of parameter space, remove the region?

We could take Hogg's idea of creating a line LIST before we start. I think this would be hard and confusing for M dwarfs. It might be better to just track lines by their central wavelength for now.

# Sub libraries

Make a series of sub-libraries from the raw files, so that they can be eaisily transferred over.

MK
KG
GF
FA

We probably don't need to bother with the parallelization stuff.

# Speedups and improvements


* Fix the mean of c0's to 1?
That way c0 is just a perturbation to the mean?
Or, should we actually be sampling in log of c0, and ensure that the mean of c0 is equal to 1?
Right now we are setting the last order to have logc0 = 0.0

* If a function requires a parameter and it's not in the parameter list, it looks up the default value.

* There should be a way to output all of the parameters into one global chain (thinned), but respecting the covariance
  between parameters and hyperparameters, if there are any.

* Introduce PSF width as a parameter that we can sample in. This would be a *correction* to the Gaussian pre-convoved
grid, which is done at something smaller that we would ever go to, for example 6.0 km/s for TRES.

#Grid Creator Code

* it turns out that the grid sampling actually needed to be *increased* to about 0.08 km/s to preserve all of the information in the raw spectrum, and InterpolatedUnivariateSpline needed to use k=5.

* specify min_vc for resampling from the HDF5 grid, as an attr for the wl file.


5188.84 is Ca II, present in Kurucz, missing in PHOENIX

Tricky PHOENIX output. Order 24 has the problem of giving substantially lower logg,
bumping up against 3.5. It seems like the multiple order sampling leaves a bi-modal distribution of parameter space,
thanks to the strong iron lines in order 24 (that are somehow incorrect). I think in this case, in particular,
some large opacity sources are present in the model that shouldn't be there, generating large, positive residuals,
which drive the fit towards lower metallicity. Labelling these as regions should really help. In fact, getting order 24
to behave might be a very good test case for the regions.


# Stellar parameter papers

* Dong 2013 LAMOST v. KIC
* Mann et al 2013, Masking out bad regions using weights that are systematically bad, and quoting intrinsic errors and systematic errors separately.
* Tkachenko: Denoising. Regions chosen due to metal lines, problems with H beta.
* Tkachenko: Spectrum analysis of Kepler target stars
* Chaplin: All kepler stars done by astroseismology. TRES script to see if these have been observed. Fundamental parameter estimation.
* Meszaros: SDSS-III APOGEE, describes APOGEE pipeline
* Kurusowa: how line profiles can trace gas
* Alaca: accretion properties using X-shooter
* Buchhave 2012 has parameters for many KOI's. We can access all the TRES data easily.
* Schornrich 2013: OSU paper for Bayseian method
* Hernandez 2004: using FAST spectra, identified spectral features sensitive to temperature
* http://arxiv.org/abs/1404.5578 Fe I oscillator strengths are off. May be why PHOENIX gives the wrong metallicity?

* Cite Barclay on GP for Kepler


# Future data and spectral libraries

* use Keck/HIRES and Keck/NIRSPEC data (search archive by name or position for our targets) GM Aur exists
http://www2.keck.hawaii.edu/koa/public/koa.php

* ELOISE spectral library
* GNIRS spectral library: http://www.gemini.edu/sciops/instruments/nearir-resources?q=node/11594
* TAPAS transmission service. Username: iczekala@cfa.harvard.edu, hertzsprung
* STScI Kurucz library

Write a download script for all the libraries
