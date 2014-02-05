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


# Object oriented rewrite

## grid_tools.py


Grid Interface (different for PHOENIX, Kurucz, BT-Settl)
* PHOENIX, Kurucz, etc inherit the Grid base class

#Master grid creation. This isn't tested, but what is probably happening is that the slowest process is actually
the writing to the file (takes about 0.25 times the "processing" of a file). The queue then becomes saturated.
Honestly, there isn't really a way around this, because there is one "master process" that is doing all the reading/writing
to the HDF5 file. Even with MPI/HDF5, this situation isn't much improved because we need to have all the attributes
updated properly.
This might actually be a hangup with the holyscratch filesystem.

#Parallel methods
*Rather than creating a map object which must pass data back and forth, really what we want is to have each process start
up and run only on it's chunk of the parameter list. That is, break up the parameter list into $nrank chunks, and process
the chunk flagged by your rank.

#rfftfreq not available in numpy 1.7 on cluster.


#Create a simple lnprob using a class and see if EMCEE still does it correctly

#Use triangle.py to make MCMC plots

#Email to DFM about using emcee with objects and class methods

#For example, this should be possible if the lnprob function did not needed to be initialized in the MPI pool to start with.


Instrument grid creation
* takes a Master HDF5 grid, and instrument object, creates a new HDF5 grid with the same attributes, does the
 interpolation, convolution, vsini, etc.
* No need to subclass? Same interface for both master file and instrument file
* Has a writer class variable that can be set, which has a write_to_FITS() method

# Test battery

Test that 0 stellar convolution works OK, for both stellar_convolve and instrument_convolve
pyFFTW wisdom and planning flags (might not be necessary for grid creation, but for actual model, perhaps)


Round out Interpolator tests

Note: if we are dealing with a ragged grid, a GridError will be raised here because a Z=+1, alpha!=0 spectrum can't be found.
We will have to simply fix the interpolation to three parameters in this case, if we think Z will be positive.
Likewise, if we think the model will be alpha enhanced, then we need to limit Z to less than 0.

Can add "skip methods" or robust=False to allow bypassing checks.

For production runs, can pre-determine wisdom for given array shapes, since they will be similar.

#Tests
* Learn how to setup sample test directories with test files

The Model and lnprob classes will require their own full suite that will be different than grid_tools

If a function requires a parameter and it's not in the parameter list, it looks up the default value.

#When will we use only three paramaters?

#The master grid (whether created from PHOENIX, Kurucz, or BT-Settl) will always have an alpha label.

#We may actually only want to interpolate in 3 values, if we are fixing alpha. This means the interpolator can be
#queried with only temp, logg, and Z.


Effbot: Two other uses are local caches/memoization; e.g.

def calculate(a, b, c, memo={}):
    try:
        value = memo[a, b, c] # return already calculated value
    except KeyError:
        value = heavy_calculation(a, b, c)
        memo[a, b, c] = value # update the memo dictionary
    return value

Memoization of python, using a decorator might be helpful, to have a dict of which grid parameters have been loaded
* https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
* can set a max number of spectra to keep in cache


### How to implement
* multiple inheritance
* composite objects
* decorator pattern is good over multiple inheritance when you want to choose between optional behaviours

# lnprob

* MCMC object could use a lnprob method or class, which takes a Model object and a Data object and compares the two
* this could also implement priors depending on the type of model and parameter lists
* fix parameters using dictionary keywords
* what about certain model specific parameters, like Chebyshev coefficients

### Data object
* contains spectrum object, which also has masks and an error spectrum
* photometry object
* also a link to the specific instrument which created it

### Model object
* contains spectrum object
* photometry object
* instrument object? do we want to fit for FWHM or kernel? What about Gauss-Hermite polynomials?
* this might require a link to the data to know exactly what wl and how many orders to downsample to
* if no data is linked to, it just outputs the raw wl
* can set degree of polynomial fits based to call to instrument. Does this also need a link to the instrument object?
Or can it use the link through the data object.

## A major problem is how to elegantly handle multiple length parameter lists
* arises for grid, lnprob, model, etc...
* could use **kwargs and then have a parameters.update() dictionary, only the parameters that are in the dictionary are
fit for, otherwise there are default values for each function?
* this looks like it will work well


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



