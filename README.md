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
* solid angle for spot coverage/accretion luminosity

"Nuisance" calibration parameters, four Chebyshev coefficients for each order. From testing, we find that three (or four) might be sufficient to encapsulate any systematic residual error from flux-calibration or blaze-removal. In any case, these corrections should be small (less than 5 percent).

Mass has a prior from the sub-millimeter dynamical mass measurement. These parameters are directly related to stellar radius R and distance d through log(g) = log (G M/R^2) and a model-to-observed flux scaling R^2/d^2. The luminosity is given by L = 4 pi F R^2, where F = f_lambda dd lambda is the bolometric flux measured at the stellar surface. Currently our method uses log (g) because this is a natural parameter of the PHOENIX model spectra grid, though a combination of the other parameters could just as easily be used. All synthetic spectra are normalized to one solar luminosity (bolometric).

## Data set

* *Spectra*: High resolution TRES spectrum. 51 orders, R = 48,000 (FWHM = 6.8 km/s) with typical signal to noise of 50 in each resolution element. All target spectra are flux-calibrated to within 10% error in c0 and 3% error in higher terms.
* *Photometry*: UBVRI, Herbst, Grankin, 2MASS, and other sources. ugriz from Keplercam. Generally, some of infrared points will be contaminated with disk emission. ugriz data from Keplercam.

# PHOENIX Models

* Investigate Lejune models
* Get BT-Settl models from France Allard's website

# TRES Data and Reduction

* Get sigma spectrum from IRAF/echelle, but I will have to do my own reduction.
* Check method against Willie2012 targets: WASP-14 and HAT-P-9
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

* use the full PHOENIX spectrum from the master grid (and interpolator), then downsample to the TRES instrument
* Finish necessary grid_tools fixes/tests in order to guarantee the Master grid is correct for the production run
* Update interpolator so that it can draw from HDF5 in parallel/mpi
* Function to TRES-convolve spectrum from grid. Maybe this is part of Model?

* sort out usage of C.kms_air and C.kms_vac
* Update FITS writer to choose the unit correctly

# Object oriented rewrite

## grid_tools.py

* script to use grid processor to generate just one FITS spectrum

pyFFTW wisdom and planning flags (might not be necessary for grid creation, but for actual model, perhaps)

Note: if we are dealing with a ragged grid, a GridError will be raised here because a Z=+1, alpha!=0 spectrum can't be found.
We will have to simply fix the interpolation to three parameters in this case, if we think Z will be positive.
Likewise, if we think the model will be alpha enhanced, then we need to limit Z to less than 0.

# Model class

* Need own LogLam spectrum object that is fast

## Sampling order designed in the lnprob directories/script

There is a Sampler object, which has an emcee object instantiated for a lnprob but with the other parameters described
by the args. Is it possible to pause a sample run and update the args? Yes, with some hacking.

1. Fix up ModelInterpolator behavior, write documentation and test suite (DONE)
2. Fix up ModelSpectrum behavior, Load spectrum, do vsini and instrument convolve properly, write documentation and test suite.
3. Figure out downsample behavior, maybe it's a component of ModelSpectrum
4. Implement Chebyshev comparions, including set_cheb methods

--- might have to break here to redo creation of master HDF5 grid, or at least start dealing with Odyssey problems ---

# Merge OOP back into Master branch.

5. Try running Sampler by alternating between stellar parameters and cheb parameters
6. Start implementing Covariance matrices


Sampler object contains

How to handle parameter lists?
* could use **kwargs and then have a parameters.update() dictionary, only the parameters that are in the dictionary are
fit for, otherwise there are default values for each function?
* this looks like it will work well

* Model object contains:

    Objects of the following

    * Data Object
        * contains spectrum object, which also has masks and an error spectrum
        * photometry object
        * also a link to the specific instrument which created it?

    * Model spectrum

        * contains spectrum object
        * photometry object
        * instrument object? do we want to fit for FWHM or kernel? What about Gauss-Hermite polynomials?
        * downsample to Data spectrum wls (when passed Data spectrum as argument, could also be part of a evaluate method)

        * Contains a ModelInterpolator object that determines how to chunk model spectrum, returns only flux. Has stored a "raw_wl" object.

    * Chebyshev coefficients
        * set cheb functions

    * Covariance matrix
        * overall poisson noise set using DataObject.sigma
        * instantiate regions
        * Keep track of regions: make sure they do not overlap
        * set individual regions? Do we want to Gibbs sample in just the individual regions?

    And has methods for manipulating the comparison of all of these based upon what is fastest.


    * Has stored a global lnprob reference for each of these sub-components and an evaluate function.

There are outside (module level) lnprob functions that reference a model object and make comparisons with individual
instance objects of it (for example, set chebyshev coefficients, evaluate).

``evaluate`` function can also be a global level function that is essentially the same for all +/-
priors for each of the parameters.

Sets of parameters describing each of these
And sampler objects that will sample them
    * (Each of these could be derived from a sampler class. They all need an "update args" method,
    a "run" method, and a pause method. The grand sampler needs specifications that it burns in between the first two,
    etc, rotates between all of the other two, etc. Some easy way to specify this behavior, whether it be lists of
    sample steps that are consumed or something else.
A general Gibbs sampler that will alternate between all of them


Drastically speed up the burn in by first sampling in the cheb components and FF, then vz and Av,
 then vsini, and then resorting to the heavier stellar parameters.

For some period of burn in, alternate between sampling

* stellar parameters, including Poisson noise factor sigma_P
* Chebyshev parameters


THEN

* Identify the pixel with the largest residual
* Instantiate a region with mu = pixel (+/- 2 or 3 pixels), a, sigma in some range
* We can have some priors on this region, ie, must remain w/in 5 AA of originally declared bad point. These priors
 are updated on the general model
* These regions can be stored as individual sparse matrices, and then final covariance matrix is a super-position of all
of them

Open questions:

* How many times should we add lines? Do you ever delete lines?
* How do you find the next-highest pixel not already covered by a region?

Code structures:

* Are parameters always referred to by their dictionary name? "temp", "logg", "Z", "vsini", etc? How is this passed
 between emcee ``p`` and the dict value? What about dropping out a value, such as Av?
* If a function requires a parameter and it's not in the parameter list, it looks up the default value.
* how are chebyshev coefficients parsed? Are these sampled in **each order** individually? Ie, their own sampler? I
think it might be cleaner to just do them all together.











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



