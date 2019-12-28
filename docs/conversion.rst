###################
Conversion to 0.3.0
###################

There have been some significant changes to *Starfish* in the upgrades to version ``0.3.0``. Below are some of the main changes, and we also recommend viewing some of the :doc:`examples/index` to get a hang for the new workflow.

.. warning::
    The current, updated code base does not have the framework for fitting multi-order Echelle spectra. We are working diligently to update the original functionality to match the updated API. For now, you will have to revert to Starfish ``0.2.0``.

.. note::
    Was there something in Starfish's utilities you used that was meaningfully removed? Open an `issue request <https://github.com/iancze/starfish/issues>`_ and we can work together to find a solution.

API-ification
=============

One of the new goals for *Starfish* was to provide a more Pythonistic approach to its framework. This means instead of using configuration files and scripts the internals for *Starfish* are laid out and leave a lot more flexibility to the end-user without  losing the functionality.

**There are no more scripts**

None of the previous scripts are included in version ``0.3.0``. Instead, the functionality of the scripts is enocded into some of the examples, which should allow users a quick way to copy-and-paste their way into a working setup.

**Analysis is made easier using other libraries**

The previous analysis code for the MCMC chains left us with a decision to make: keep it baked in and locked to an exact MCMC library (*emcee*) or remove it from the project and let other libraries handle it. We chose the latter. Our recommendations for analyzing Bayesian MCMC chains is `arviz <https://arviz-devs.github.io/arviz/>`_.

**There is no more config.yaml**

This file has been eliminated as a byproduct of two endeavors: first is the elimination of the scripts- with a more interactive API in mind, we don't need to hardcode our values in a configuration file. Second is the smoothing of the consistency between the grid tools, the spectral emulator, and the statistical models. For instance, we don't need a configuration value for the grid parameter names because we can leave these as attributes in our GridInterfaces and propagate them upwards through the classes that use the interface. 

**The modularity has skyrocketed**

One of the BIGGEST products of this rewrite is the simplification of the core of what *Starfish* provides: a statistical model for stellar spectra. If you have extra science you want to do, for example: binary star modelling, debris disk modeling, sun spot modeling, etc. we no longer lock down the full maximum likelihood estimation process. Because the new models provide, essentially, transformed stellar models and covariances, if we want to do our own science with the models beyond what *Starfish* already does, we can just plug-and-play! Here is some psuedo-code that exemplifies this behavior:

.. code-block:: python

    from Starfish.models import SpectrumModel
    from Starfish.emulator import Emulator
    from astropy.modeling import blackbody
    
    emu = Emulator.load('emu.hdf5')
    model = SpectrumModel(..., **initial_parameters)

    flux, cov = model()
    dust_flux = blackbody(model.data.waves, T_dust)
    flux += dust_flux

    # Continue with MLE using this composite flux

Overall, there are a lot of changes to the workflow for *Starfish*, too. So, again, I highly recommend looking through some :doc:`examples/index` and browsing through the :doc:`api/index`. 



Maintenenance
=============

**Clean up**

Much of the bloat of the previous repository has been pruned. There still exists archived versions from the GitHub releases, but we've really tried to turn this into a much more professional-looking repository. If there were old files you were using or need to have a copy of, check out the archive.

**CI Improvements**

The continuous integration has also been improved to help limit the bugs we let through as well as vamp up some of the software development tools that are available to us. You'll see a variety of more pretty badges as well as a much-improved travis-ci matrix that allows us to test on multiple platforms and for multiple python versions

**Cleaning up old Issues**

Many issues are well outdated and will soon become irrelevant with version ``0.3.0``. In an effort to remove some of the clutter we will be closing all issues older than 6 months old or that are solved with the new version. If you had an old issue and feel it was not resolved, feel free to reach out and reopen it so we can work on further improving *Starfish*. 
