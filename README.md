StellarSpectra
==============

Repository for MCMC fitting of TRES T Tauri Spectra

Copyright Ian Czekala 2013 (pronounced Check-al-uh, or Check-aw-ah, if you want to be Polish about it.)

`iczekala@cfa.harvard.edu`

# Using git to interact with this repository

This is a test.

First, if you have not already done so, create a github [user account](https://github.com/) and [install git](http://git-scm.com/downloads) on your computer.

If you are at the CfA and using the CF, you can use the version of git installed by Tom Aldcroft by adding the following to your ``.cshrc``

    set path=(/data/astropy/ska/arch/x86_64-linux_CentOS-5/bin $path)

You may also want to set up your [SSH keys](https://help.github.com/articles/generating-ssh-keys).

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

Generally, what these documents boil down to is that if some time has elapsed and there are updates to the repository on github, before doing ``git push``, you must first update your local copy by ``cd``ing to the StellarSpectra directory and executing

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

We estimate stellar parameters by taking a Bayesian approach to generate a model of the data (and systematic errors) and sample this posterior using the `emcee` implementation of the Markov Chain Monte Carlo ensemble sampler by Goodman and Weare (2010).

# Installation

Currently, StellarSpectra has many dependencies, however most of them should be satisfied by an up-to-date scientific
python installation. We highly recommend using the
[Anaconda Scientific Python Distribution](https://store.continuum.io/cshop/anaconda/) and updating to python 3.3.

StellarSpectra requires the following packages:

* numpy†
* scipy†
* jinja2†
* matplotlib†
* h5py†
* astropy†
* cython†
* pyyaml†
* emcee
* pyfftw

Those marked with a † are included in the latest Anaconda distribution. The others should be available from ``pip`` or from the
project websites.

And the following library

* the [SuiteSparse](https://www.cise.ufl.edu/research/sparse/SuiteSparse/) sparse matrix library, which is written in C.

First recommend installing SuiteSparse. If you have administrator priveledges, installing from a package manager is
 likeley the easiest. Otherwise, there is always downloading the package and installing from source to a
 user defined directory.

You can change the install directory by viewing the README.txt, and then editing SuiteSparse_config.mk

For example, if you installed SuiteSparse to mylib, then you would call setup.py as

    python setup.py build_ext --inplace -Lmydir/lib -Imydir/include

For example, to use my SuiteSparse directories on the CfA CF network,

    python setup.py build_ext --inplace -L/pool/scout0/.build/lib -I/pool/scout0/.build/include


To build

    $ python setup.py build_ext --inplace

If you have control of your own system

    $ sudo python setup.py develop

If not, (for example on the CF network)

    $ python setup.py develop --user

## Parameters

Model parameters

* [Fe/H]: metallicity, allowed to vary between -0.5 and +0.5
* T_eff: effective temperature of photosphere
* log g: surface gravity
* v sin i: for rotational broadening
* v_z: radial velocity
* Av: extinction
* solid angle for spot coverage/accretion luminosity

"Nuisance" calibration parameters, four Chebyshev coefficients for each order. From testing, we find that three (or four) might be sufficient to encapsulate any systematic residual error from flux-calibration or blaze-removal. In any case, these corrections should be small (less than 5 percent).

Mass has a prior from the sub-millimeter dynamical mass measurement. These parameters are directly related to stellar radius R and distance d through log(g) = log (G M/R^2) and a model-to-observed flux scaling R^2/d^2. The luminosity is given by L = 4 pi F R^2, where F = f_lambda dd lambda is the bolometric flux measured at the stellar surface. Currently our method uses log (g) because this is a natural parameter of the PHOENIX model spectra grid, though a combination of the other parameters could just as easily be used. All synthetic spectra are normalized to one solar luminosity (bolometric).

## Data set

* *Spectra*: High resolution TRES spectrum. 51 orders, R = 48,000 (FWHM = 6.8 km/s) with typical signal to noise of 50 in each resolution element. All target spectra are flux-calibrated to within 10% error in c0 and 3% error in higher terms.
* *Photometry*: UBVRI, Herbst, Grankin, 2MASS, and other sources. ugriz from Keplercam. Generally, some of infrared points will be contaminated with disk emission. ugriz data from Keplercam.


* Search TRES archive by position for all stars in proposal

# Code

## Operation modes

* Sampling in all c for each order
* marginalization only: regeneration of conditionals after the fact
* using an un-flux calibrated TRES spectrum. Is this as simple as not putting a prior on c0?
* operate on just one order, for all orders using a JobArray


## Synthetic photometry comparison

* implement it on the full spectrum
* can add in extra flux for the H alpha line.

## Mass prior

* for a 5% measurement in mass, we improve T_eff, log g, R estimates by XX amount.

## Desireable plotting tools and output analysis

* color code random samples by lnprob value?
* Easy staircase plot generator (partially done)
* use autocorrelation time, and plots for each walker
* Jinja2 output: multithreading for quicker plotting
* create my own "linelist" using an astropy table, easily plot line regions of data in a "gallery"
* change masking list into an astropy table
* easily "jump around" in parameters, ie, automatically solve for nuisance coeffs while jumping around in T, log g, Z. Interactive.

## Error analysis

* This would be good to do first to determine a statistical uncertainty floor
* Find regions of spectrum that are poorly modelled after Mann 2013, comparison with many different best-fit spectra.
* Histograms of residuals of fits, tests to determine if Gaussian, Lorentzian, or student-t distributions would be the best fit
* color-code the level of sigma that is due to each level of error: Poisson, systematic, flux-calibration
* determine an 'interpolation error spectrum' 
* Complete prior error determination using `test_priors.py` and write this up in research notebook. Also update paper.
* Determine high and low information content regions of the spectrum.
* also fit sky spectrum, with no wavelength shift

## Potential Speedups

* Now that I am interpolating in T_eff, log g space, convert fitting to use mass, radius, distance, and luminosity. 
* Be so bold as to interpolate the Fourier transform?
* Fine tune garbage collection (debugging?): http://docs.python.org/2/library/gc.html#gc.garbage
* Ask Paul Edmon if I can write to /n/scratch2
* tweak grid generation mechanism so that it doesn't fill up memory (try removing list()). Output only the error messages.

# Method Checks and concerns 

* Flux interpolation errors (a systematic spectrum) (see notes + figure from 9/9/13)
* distribution of interpolation errors (histogram)
* What is a veiling prior? Hartigan 1991
* Make mini_sampler pickleable following emcee

# Beyond TRES

* use Keck/HIRES and Keck/NIRSPEC data (search archive by name or position for our targets) GM Aur exists
http://www2.keck.hawaii.edu/koa/public/koa.php
* Make a tool to collect all the targets into an astrotable that is easily sortable and can query to output targets in RA/DEC and move easily between them.
* For knowledge purposes, try synthesizing a spectrum using the model atmosphere provided. This might be harder than I currently imagine.
* ELOISE spectral library
* GNIRS spectral library: http://www.gemini.edu/sciops/instruments/nearir-resources?q=node/11594
* TAPAS transmission service. Username: iczekala@cfa.harvard.edu, hertzsprung

# MISC

* What about using PyMC: http://pymc-devs.github.io/pymc/
* Read about index arrays: http://docs.scipy.org/doc/numpy/user/basics.indexing.html
* try FFTW

Crop out edges from zero response for TRES data
    
	imcopy GWOri.fits[6:2304,*] GWOri_crop.fits


* astropy echelle reader tool: properly reading in Chebyshev fit (for astropy) http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?specwcs
* an animated diagram of age tracks in the HR diagram for my talk? Put on webpage?

## How to use memory_profiler
Put the decorator `@profile` over the function you want to profile

	python -m memory_profiler model.py

/scratch is about twice as fast as /holyscratch

# Roadmap

* sort out usage of C.kms_air and C.kms_vac

## grid_tools.py

alpha yes/no needs to be determined upon initialization.

# Covariance and SparseMatrices

* convert covariance function to work on velocities instead of wavelengths

Order 24 has the problem of giving substantially lower logg, bumping up against 3.5. It seems like the
multiple order sampling leaves a bi-modal distribution of parameter space, thanks to the strong
iron lines in order 24 (that are somehow incorrect).

## Line logic

* How many times should we add lines? Do you ever delete lines?

Is it a problem when the sigma's overlap? Can we better insure the tapering?

Should we make some assertion that the amplitude of the region can't be less than the global covariance height?
Otherwise in the cleanup logic, we remove the region?

# Speedups and improvements

JSON serialization
Always output the model position on finish, so we can resume sampling if need be

* Output, plot the sigma_matrix as a 1D slice through row=constant (how correlated is a given pixel with everything else?)
* Could we scroll through rows?

* Visualize cheb functions
* Visualize spectrum and residuals
* Serialize the model state using JSON
* Load parameter files for runs using JSON (might be nice for creating visualizations, too)
* Write samples in flatchain to an HDF5 file
* Extend config files for # of samples

* Fix the mean of c0's to 1?
That way c0 is just a perturbation to the mean?
Or, should we actually be sampling in log of c0, and ensure that the mean of c0 is equal to 1?
Right now we are setting the last order to have logc0 = 0.0


* If a function requires a parameter and it's not in the parameter list, it looks up the default value.



#Grid Creator Code

* it turns out that the grid sampling actually needed to be *increased* to about 0.08 km/s to preserve all of the information in the raw spectrum, and InterpolatedUnivariateSpline needed to use k=5.

* from the finished product, chunking loading is used 400:500 instead of [False False True True ... False] ? needs testing/timing.
* interpolator combines the averaged spectra
* for instrument and stellar convolve, the spectra are first resampled to a finer grid, then FFT'ed, then downsampled.
* chunking will be different. instead, it chunks #points over a set wl range

* if things are really slow this way, we can go back to the old way. First, create a master grid that is log-lam spaced for
just the wl ranges, then do the MCMC run on it.

* specify min_vc for resampling from the HDF5 grid, as an attr for the wl file.

* If ModelSpectrum.downsample() is too slow because of too many points, we can do the intermediate step of reducing the
sampling after the FFT. Probably can speed up by 30% or so with this change. The first Interpolation to the FFT grid
is not that slow, but it's the interpolation from the FFT grid to the Data grid that is slow. This is probably
because a lot of evaluation at all the points is needed. So by reducing the number of points we'll be playing it safe.

* ModelInterpolater._determine_chunk() will need to be updated to use create_log_lam_grid()

# 5188.84 is Ca II, present in Kurucz, missing in PHOENIX


# Running the model (to be moved to the docs?)


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


# Process Kurucz grid using grid creator code


# Interesting Paper points

    1. the covariance methodology applied to this case (and why);
    2. the power of the technique in terms of modularity and a proper probabilistic treatment with great flexibility.
    I think we want to demonstrate that it works in a few key examples, maybe:
        a. some TRES/Kepler spectra to compare with previous results from SPC, SME, etc.;
        b. some super-obvious or over-constrained case (e.g., Vega, or maybe some Kepler source with astroseismology); and
        c. maybe some tricky case to demonstrate flexibility, e.g., a composite spectrum from a double-lined
        spectroscopic binary?  We can worry about the examples later, since you have a lot of work to do in writing
        the basic methodology parts.

# Before 1.0 release

* update ModelSpectrum to draw defaults from C.var_defaults
* More tests for ModelSpectrum.downsample()
* cprofile lnprob, check to see if pyFFTW wisdom and planning flags would speed things up.

* Testing:
for some reason, running all of the tests in ``test_spectrum.py`` at once will fail for ModelSpectrum, but if I run
just ModelSpectrum individually then everything is fine.
