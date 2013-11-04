StellarSpectra
==============

Repository for MCMC fitting of TRES T Tauri Spectra

Copyright Ian Czekala 2013

`iczekala@cfa.harvard.edu`

# Summary of Approach

We estimate stellar parameters by taking a Bayesian approach to generate a model of the data (and systematic errors) and sample this posterior using the `emcee` implementation of the Markov Chain Monte Carlo ensemble sampler by Goodman and Weare (2010).

## Parameters

Model parameters

* [Fe/H]: metallicity, allowed to vary between -0.5 and +0.5
* T_eff: effective temperature of photosphere
* log g: surface gravity
* v sin i: for rotational broadening
* v_z: radial velocity
* Av: extinction

"Nuisance" calibration parameters, four Chebyshev coefficients for each order. From testing, we find that three (or four) might be sufficient to encapsulate any systematic residual error from flux-calibration or blaze-removal. In any case, these corrections should be small (less than 5 percent).

Mass has a prior from the sub-millimeter dynamical mass measurement. These parameters are directly related to stellar radius R and distance d through log(g) = log (G M/R^2) and a model-to-observed flux scaling R^2/d^2. The luminosity is given by L = 4 pi F R^2, where F = f_lambda dd lambda is the bolometric flux measured at the stellar surface. Currently our method uses log (g) because this is a natural parameter of the PHOENIX model spectra grid, though a combination of the other parameters could just as easily be used. All synthetic spectra are normalized to one solar luminosity (bolometric).

## Data set

* *Spectra*: High resolution TRES spectrum. 51 orders, R = 48,000 (FWHM = 6.7 km/s) with typical signal to noise of 50 in each resolution element. All target spectra are flux-calibrated to within 10% error in c0 and 3% error in higher terms.
* *Photometry*: UBVRI, Herbst, Grankin, 2MASS, and other sources. ugriz from Keplercam. Generally, some of infrared points will be contaminated with disk emission. ugriz data from Keplercam.

# PHOENIX Models

* Normalize all models to 1 solar luminosity
* Investigate Lejune models
* Get BT-Settl models from France Allard's website

# TRES Data and Reduction

* Get sigma spectrum from IRAF/echelle, but I will have to do my own reduction.
* Check method against Willie2012 targets: WASP-14 and HAT-P-9
* Search TRES archive by position for all stars in proposal

# Code

## Operation modes

* Sampling in all c for each order

	* this means we need to come up with a method to start MCMC in the minimum of each c0, and to properly set up emcee and priors. 

* marginalization only
 
	* this needs a way to determine the conditionals after one is done
	* needs tests against the full sampling in all c

* using an un-flux calibrated TRES spectrum. Is this as simple as not putting a prior on c0? 



## Synthetic photometry comparison

* implement it on the full spectrum
* can add in extra flux for the H alpha line.

## Mass prior

* for a 5% measurement in mass, we improve T_eff, log g, R estimates by XX amount.

## Fits to metallicity

* download grid, rewrite load_flux routine
* remake HDF5 grid, upload to Odyssey. 



## Desireable plotting tools and output analysis

* Drawing random samples from the data (color code by lnprob value?)
* Posterior predictive check: does the model match the data in a qualitative sense? Needs to be a better tool for doing this for multiple orders. Can you select which orders to do this for? Plot three orders at once? Reasonably show parameter values, lnprob value.
* better way to visualize posteriors on Chebyshev coefficients
* way to easily visualize outliers --> where is most of the chi^2 coming from?
* Easy staircase plot generator (partially done)
* use autocorrelation time, and plots for each walker
* automatically generate outputs of histograms and charts, summarized in a directory that is unique for each run.

## Error analysis

* Generate fake spectrum by adding in random Gaussian noise following 10/31/13. This would be good to do first to determine a statistical uncertainty floor
* Find regions of spectrum that are poorly modelled after Mann 2013, comparison with many different best-fit spectra.
* Histograms of residuals of fits, tests to determine if Gaussian, Lorentzian, or student-t distributions would be the best fit
* color-code the level of sigma that is due to each level of error: Poisson, systematic, flux-calibration
* determine an 'interpolation error spectrum' 
* Complete prior error determination using `test_priors.py` and write this up in research notebook. Also update paper.
* Determine high and low information content regions of the spectrum.


## Functionality improvements

* Better configuration files on Odyssey, launching multiple jobs at once: .yaml files.

## Potential Speedups

* Now that I am interpolating in T_eff, log g space, convert fitting to use mass, radius, distance, and luminosity. 
* Be so bold as to interpolate the Fourier transform?
* Fine tune garbage collection (debugging?): http://docs.python.org/2/library/gc.html#gc.garbage
* Option to limit range of data FFT'ing or loading (via interpolator?) when doing less than all the orders. Option to flux-interpolator?
* Move back to pyFFTW

# Method Checks and concerns 

* Flux interpolation errors (a systematic spectrum) (see notes + figure from 9/9/13)
* distribution of interpolation errors (histogram)
* What is a veiling prior? Hartigan 1991
* Try seeding random spectra (derived from models) to see if parameters are recovered at various S/N via Doppman and Jaffey, sec 4.1

# Alternative methods for normalization

* actually sample c0, c1, c2, c3 for each and every order (start walkers at the coeffs that maximize chi^2 for each order
* do this but multiplying the data and sigma, not the model
* do c0 ( 1 + c1 + c2 + c3), marginalize over all
* do c0 ( 1 + c1 + c2 + c3), marginalize over c1+, sample in c0
* do c0, but in a hierarchical fashion.
* from the final parameters, create an inverse flux calibration.

# Checks against real data

* Empirical spectral libraries: ELOISE?
* Torres 12 paper: HAT-P-9, WASP-14
* Real data in HIRES archive from George Herbig

# Beyond TRES

* use Keck/HIRES and Keck/NIRSPEC data (search archive by name or position for our targets) GM Aur exists
http://www2.keck.hawaii.edu/koa/public/koa.php
* Make a tool to collect all the targets into an astrotable that is easily sortable and can query to output targets in RA/DEC and move easily between them.
* For knowledge purposes, try synthesizing a spectrum using the model atmosphere provided. This might be harder than I currently imagine.

# MISC

* What about using PyMC: http://pymc-devs.github.io/pymc/
* Read about index arrays: http://docs.scipy.org/doc/numpy/user/basics.indexing.html
* try FFTW

Crop out edges from zero response for TRES data
    
	imcopy GWOri.fits[6:2304,*] GWOri_crop.fits


* astropy echelle reader tool: properly reading in Chebyshev fit (for astropy) http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?specwcs

## How to use memory_profiler
Put the decorator `@profile` over the function you want to profile

	python -m memory_profiler model.py
