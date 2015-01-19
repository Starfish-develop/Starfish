=======
Scripts
=======

StellarSpectra is accompanied by a number of helper scripts that can be useful for formatting and exploration of the
data.

You may want to add the following location to your `PATH` in order to make frequent use of these scripts.

HDF5 Scripts
============

* hdfcat.py
* hdfplot.py

hdfcat.py
---------

This script is used to concatenate the samples from multiple MCMC runs. The output must be stored in the same formats,
but it is primarily for concatenating the samples for the stellar parameters and order parameters, but notably *not*
the region lists and parameters. This is designed to enable parallel runs of the algorithm but then combine all of
the parameters into one HDF5 file for use with `hdfplot.py`

hdfplot.py
----------

This script is designed to plot the chain data contained in a single HDF5 file. As of now, you can select which
stellar parameters you wish to provide in the triangle plot. This is designed to make custom plots *after* the run, such
that you can play around with intercomparing parameters and making it look good for the paper or presentation.

Functionality that's coming: ability to select/truncate samples from the beginning/end of the run.

hdfinterplot.py
---------------

This script is designed to plot the different chains for single parameters. It hasn't been written yet.

Plotting Scripts
================

* plotly_model.py

plotly_model.py
---------------

This script is designed to visualize an individual order. Designed to be run both as a command line script and
as an importable function that returns a url.
