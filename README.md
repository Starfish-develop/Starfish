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

### How to bin/downsample?

* np.convolve with a set of np.ones()/len(N)--> running boxcar average. However, this assumes a fixed integer width, which does not correspond to a fixed dispersion width, since there is a break at 5000 AA. However we could use different filters on either side of this break if we wanted. Is the instrumental response simply a Gaussian kernel with a width of R~48,000? (6.7 km/s)?
* Fourier methods

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
* astropy echelle reader tool

# Code

## Synthetic photometry comparison

* implement it on the full spectrum

## Fitting multiple orders

* Fit Av
* Constraints on individual C_x due to overlap?
* Bascially, need an easily configurable plotting/running system to select between how many orders to run.


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

* Easy way to set priors on Chebyshev coefficients
* Easy way to initialize Chebyshev walker positions
* Better configuration files on Odyssey, launching multiple jobs at once: .yaml files.

## Potential Speedups

* Pre-compute redenning curve at locations of Phoenix model or TRES (doesn't really matter, actually, since a delta v_z shift is at worst going to correspend to a fraction of an angstrom!
* Precompute PHOENIX grid to TRES wavepoints? I think we'd still need to do an interpolation.
* Now that I am interpolating in T_eff, log g space, convert fitting to use mass, radius, distance, and luminosity. 
* Rewrite wechelletxt (or a derivative of rechelletxtflatnpy) to write a numpy array directly (rather than loading text files into a numpy array structure, just load this directly)
* streamline multi-order as a vectorized operation (or rather, rewrite broadening and other operations to take place globally rather than per-order, and only leave the final downsample routine for per-order).
* Fourier transform methods, interpolation methods, for downgrading spectra (ie, convolve with a boxcar, to do sinc interpolation)
* Linearize the PHOENIX model to do a FFT by interpolating with zeros?

## Astropy integration

* properly reading in Chebyshev fit (for astropy) http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?specwcs

## Code concerns

* Pixel spacing changes at 5000 Ang in the PHOENIX model, we need to change the kernel size for the convolution. How much does the kernel actually change across an order? Maybe we can do this as an FFT convolve with the whole spectrum? Likewise, we could 

# Method Checks and concerns 

* What if we had a single long-slit spectrum, how should we do the convolution?
* Edge effects with convolution
* Flux interpolation errors (a systematic spectrum) (see notes + figure from 9/9/13)
* distribution of interpolation errors (histogram)
* Are my errors done correctly (weight by blaze?) or should weight by inverse pixel count
* What is a veiling prior? Hartigan 1991
* use student t distribution to give less weight to outliers that may be distorting the fit?

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



