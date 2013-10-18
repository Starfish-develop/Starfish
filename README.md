StellarSpectra
==============

Repository for MCMC fitting of TRES T Tauri Spectra

Copyright Ian Czekala 2013

`iczekala@cfa.harvard.edu`

# Summary of Approach

We estimate stellar parameters by taking a Bayesian approach to generate a model of the data (and systematic errors) and sample this posterior using the `emcee` implementation of the Markov Chain Monte Carlo ensemble sampler by Goodman and Weare (2010).

## Parameters

Model parameters

* [Fe/H]: metallicity, fixed to solar
* T_eff: effective temperature of photosphere
* log g: surface gravity
* v sin i: for rotational broadening
* v_z: radial velocity
* Av: extinction

"Nuisance" calibration parameters, three Chebyshev coefficients for each order, although this could be arbitarily expanded to more. From testing, we find that three (or four) might be sufficient to encapsulate any systematic residual error from flux-calibration or blaze-removal. In any case, these corrections should be small (less than 5 percent).

Mass has a prior from the sub-millimeter dynamical mass measurement. These parameters are directly related to stellar radius R and distance d through log(g) = log (G M/R^2) and a model-to-observed flux scaling R^2/d^2. The luminosity is given by L = 4 pi F R^2, where F = f_lambda dd lambda is the bolometric flux measured at the stellar surface. Currently our method uses log (g) because this is a natural parameter of the PHOENIX model spectra grid, though a combination of the other parameters could just as easily be used

## Data set

* *Spectra*: High resolution TRES spectrum. 51 orders, R = 48,000 (FWHM = 6.7 km/s) with typical signal to noise of 50 in each resolution element. 
* *Photometry*: UBVRI, Herbst, Grankin, 2MASS, and other sources. ugriz from Keplercam. Generally, some of infrared points will be contaminated with disk emission.

# PHOENIX Models

* Normalize all models to 1 solar luminosity
* Investigate Lejune models
* Get BT-Settl models from France Allard's website

# TRES Data and Reduction

Crop out edges from zero response
    
	imcopy GWOri.fits[6:2304,*] GWOri_crop.fits

* Rather than weight by inverse blaze, weight by inverse photon counts?
* Weight by the sigma spectrum? (will have to do own reduction)
* Get sigma spectrum from IRAF/echelle
* Check method against Willie2012 targets: WASP-14 and HAT-P-9
* Read in general spectra (both flux calibrated and un-flux calibrated)
* Be able to deal with un-flux calibrated data from different instruments (ie, TRES)
* Search TRES archive by position for all stars in proposal
* astropy echelle reader tool: properly reading in Chebyshev fit (for astropy) http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?specwcs

# Code

## Synthetic photometry comparison

* implement it on the full spectrum

## Fitting multiple orders

* Fit Av, create samples at TRES grid positions
* Constraints on individual C_x due to overlap?
* Bascially, need an easily configurable plotting/running system to select between how many orders to run.
* Easy way to set priors on Chebyshev coefficients
* Easy way to initialize Chebyshev walker positions


## How to use memory_profiler
Put the decorator `@profile` over the function you want to profile

	python -m memory_profiler model.py


## Desireable plotting tools and output analysis

* Drawing random samples from the data (color code by lnprob value?)
* Posterior predictive check: does the model match the data in a qualitative sense? Needs to be a better tool for doing this for multiple orders. Can you select which orders to do this for? Plot three orders at once? Reasonably show parameter values, lnprob value.
* better way to visualize posteriors on Chebyshev coefficients
* way to easily visualize outliers --> where is most of the chi^2 coming from?
* Easy staircase plot generator (partially done)
* use autocorrelation time
* run multiple chains

## Functionality improvements

* Better configuration files on Odyssey, launching multiple jobs at once: .yaml files.

## Potential Speedups

* Now that I am interpolating in T_eff, log g space, convert fitting to use mass, radius, distance, and luminosity. 
* Be so bold as to interpolate the Fourier transform?
* Fine tune garbage collection (debugging?): http://docs.python.org/2/library/gc.html#gc.garbage
* Option to limit range of data FFT'ing or loading (via interpolator?) when doing less than all the orders. Option to flux-interpolator?

## Code concerns

* What does frequency response of linear interpolation do to the spectrum? How bad is it?
* Conserve equal signal to noise per resolution element? (Or the same S/N per resolution element as before). (Deal with this when we get our data).

# Method Checks and concerns 

* Flux interpolation errors (a systematic spectrum) (see notes + figure from 9/9/13)
* distribution of interpolation errors (histogram)
* Are my errors done correctly (weight by blaze?) or should weight by inverse pixel count
* What is a veiling prior? Hartigan 1991
* use student t distribution to give less weight to outliers that may be distorting the fit?
* Try seeding random spectra (derived from models) to see if parameters are recovered at various S/N via Doppman and Jaffey, sec 4.1

# Alternative methods for normalization

* actually sample c0, c1, c2, c3 for each and every order (start walkers at the coeffs that maximize chi^2 for each order
* do this but multiplying the data and sigma, not the model
* do c0 ( 1 + c1 + c2 + c3), marginalize over all
* do c0 ( 1 + c1 + c2 + c3), marginalize over c1+, sample in c0
* do c0, but in a hierarchical fashion.

# Checks against real data

* Empirical spectral libraries: ELOISE?
* Torres 12 paper: HAT-P-9, WASP-14
* Real data in HIRES archive from George Herbig

# Flux Calibration package

* Get Bayesian hierarchical inference (or just a prior on systematic calibration (how tight should my priors on the Chebyshev coefficients be?)
* Get "sigma spectrum"
* Remedy blaze function removal

# Beyond TRES

* use Keck/HIRES and Keck/NIRSPEC data (search archive by name or position for our targets) GM Aur exists
http://www2.keck.hawaii.edu/koa/public/koa.php
* Make a tool to collect all the targets into an astrotable that is easily sortable and can query to output targets in RA/DEC and move easily between them.

# MISC

* What about using PyMC: http://pymc-devs.github.io/pymc/
* Setting up Gibbs sampling for coefficients
* Read about index arrays: http://docs.scipy.org/doc/numpy/user/basics.indexing.html
* try FFTW
