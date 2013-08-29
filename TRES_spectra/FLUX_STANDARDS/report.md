What have I tried with flux calibration?

Reading in high and low resolution bandpass calibrators

Issues with high resolution (R 50 000): 

* simply too many points to deal with, suffer from low s/n
* overconstrained fitting

Attempt at moderate resolution (5 Ang)

* works well to fit a 3rd order polynomial 
* need to delete points around spectra lines, these are usually the ones with the most RMS
* the final spectrum looks fairly decent, yet does have some residual wiggles
* it would be great to compare these against a real stellar spectrum, for which we have a model.


Future improvements

* There are many orders that overlap, that dont have a spectral line, and are shared among many standards.

Plot all orders together, spectra calibrated by other sensfiles, see how well this looks.

Know that the edges of the flux should overlap, perhaps this is a way to rectify the curves


Really need a quick routine to read in echelle spectra to python




