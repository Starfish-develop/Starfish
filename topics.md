# Stellar Spectra

## Parameters

Grid parameters

* [Fe/H]
* T_eff
* log g
* v sin i for rotational broadening


## PHOENIX Models
Idea is to match TRES observations as best as possible. Many tasks to be done, but in what order?

1. Normalize to continuum
2. Rotational broadening profile
3. Instrumental response function
4. Binning (or sample, at) central wavelenth to central wavelength? 

### Data concerns

* May need to use polynomials to do sampling/interpolation

### How to bin?

* np.convolve with a set of np.ones()/len(N)--> running boxcar average. However, this assumes a fixed integer width, which does not correspond to a fixed dispersion width, since there is a break at 5000 AA. However we could use different filters on either side of this break if we wanted. Is the instrumental response simply a Gaussian kernel with a width of R~44,000?


## TRES Reduction

Issues with GW Ori spectrum

* Not flux calibrated
* What is y-axis? Photon counts?

What to do with our data

* Crop out edges 
	
	--> imcopy GWOri.fits[6:2304,*] GWOri_crop.fits

* Properly normalize to the continuum (S/N issues for overlap?)

### Combine spectra into 1D

* `scombine` interpolation works only with flux-calibrated data. Does not preserve count levels.

* Mask out emission lines 

How does flux calibration affect the comparison against a normalized continuum?

* properly reading in Chebyshev fit 

http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?specwcs

## Odyssey

* How much space do we have? 20G
* Download models to directory?
* Queue to submit jobs to? There seem to be some general purpose queues. 
* Interpolate and then bin? How fast are the operations?


# Continuum fitting and normalization

* wavelet transforms
* potential to mask out bad regions
