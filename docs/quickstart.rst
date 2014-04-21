==========
Quickstart
==========

Downloading the program data
============================

In order to fit the data, you will need to download libraries of synthetic spectra. For those on the CfA network,
 I have stored many of these synthetic libraries at ``/pool/scout0/libraries/*.hdf5``. Many of these might be quite large,
 so make sure you have enough space (libraries range from 10 - 100 Gb).

More information about how to download raw spectra and create your own grids is available in :doc:`grid_tools`.

Creating the parameter script
=============================

All parameters are specified in YAML. The specification is [here](http://www.yaml.org/spec/1.2/spec.html) and the
 python interface is documented [here](http://pyyaml.org/wiki/PyYAMLDocumentation). YAML is a very powerful configuration
 language but it's also very easy to use, so don't be intimidated. It's a highly worthwhile format to learn,
 and you can probably figure out how it works from the example scripts.

Additionally, you can also configure the fitting runs by directly editing the python script ``scripts/stars/base_lnrob``
If you wanted to, you could also remove the YAML dependencey and just declare your variables in the script, but I think
it is nice to keep them separate for organization's sake.


Echelle orders
==============

To prevent confusion, all echelle orders will be indexed from 0. This means that if there are :math:`N` total orders, then
 the first order is 0 and the last order is :math:`N - 1`.


Covariance matrices
===================

Useful link to Wikipedia on covariance matrices/positive-semi definite