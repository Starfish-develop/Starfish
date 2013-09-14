# Stellar Spectra

## Parameters

Model parameters

* [Fe/H] -> fixed to solar
* T_eff
* log g
* v sin i for rotational broadening
* v_z radial velocity
* Av extinction

## PHOENIX Models

### How to bin?

* np.convolve with a set of np.ones()/len(N)--> running boxcar average. However, this assumes a fixed integer width, which does not correspond to a fixed dispersion width, since there is a break at 5000 AA. However we could use different filters on either side of this break if we wanted. Is the instrumental response simply a Gaussian kernel with a width of R~44,000? (6.7 km/s)?


## TRES Reduction

* Crop out edges 
	--> imcopy GWOri.fits[6:2304,*] GWOri_crop.fits

* Rather than weight by inverse blaze, weight by inverse photon counts?
* Weight by the sigma spectrum? (will have to do own reduction)


## Fitting multiple orders

* Fit Av
* Constraints on individual C_x due to overlap?


## Data side
Get sigma spectrum from IRAF/echelle

Check method against Willie2012 targets: WASP-14 and HAT-P-9
Read in general spectra (both flux calibrated and un-flux calibrated)
Be able to deal with un-flux calibrated data from different instruments (ie, TRES)


# How to use memory_profiler
python -m memory_profiler model.py
then just put the decorator @profile over the function you want to profile


* properly reading in Chebyshev fit (for astropy)
http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?specwcs
