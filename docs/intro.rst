============
Introduction
============

Deriving Physical Parameters from Astronomical Spectra
======================================================

**Consider the following scenario**: An astronomer returns from a successful observing trip with many high signal-to-noise,
high resolution stellar spectra on her trusty USB thumbdrive. If she was observing with the type of echelle spectrograph
common to most modern observatories, chances are that her data span a significant spectral range, perhaps the full optical (
3700 to 9000 angstrom) or the full near-infrared (0.7 to 5 microns). Now that she's back at her home institution,
she sets herself to the task of determining the stellar properties of her targets.

She is an expert in the many existing well-tested techniques for determining stellar properties, such as
`MOOG <http://www.as.utexas.edu/~chris/moog.html>`_ and `SME <http://www.stsci.edu/~valenti/sme.html>`_.
But the fact that these use only a small portion of her data—several well-chosen lines like Fe and Na—has stubbornly persisted
in the back of her mind.

At the same time, the astronomer has been paying attention to the steady increase in availability of high quality synthetic
spectra, produced by a variety of groups around the world. These libraries span a large range of the stellar parameters
she cares about (effective temperature, surface gravity, and metallicity) with a tremendous spectral coverage from the
UV to the near-infrared—fully covering her dataset. She wonders, "instead of choosing a subset of lines to study, what
if I use these synthetic libraries to fit *all of my data*?"

**She knows that it's not quite as simple as just fitting more spectral range**. She knows that even though
the synthetic spectral libraries are generally high quality and quite remarkable in their scope, it is still very hard
to produce perfect synthetic spectra. This is primarily due to inaccuracies in atomic and molecular constants that are
difficult to measure in the lab, making it difficult to ensure that all spectral lines are accurate over a wide swath
of both stellar parameters and spectral range. The highest quality libraries tend to achieve their precision by
focusing on a "sweet spot" of stellar parameters near those of the Sun, and by choosing a limited spectral range,
where atomic constants can be meticulously vetted for accuracy.

The astronomer also knows that some of her stars may
have non-solar ratios of elemental abundances, a behavior that is not captured by the limited set of adjustable parameters t
hat specify a spectrum in a synthetic library. She's tried fitting the full spectrum of her stars using a simple :math:`\chi^2`
likelihood function, but she knows that ignoring these effects will lead to parameter estimates that are biased and
have unrealistically small uncertainties. She wonders, "How can I fit my entire spectrum but avoid these pitfalls?"

Introducing *Starfish*: a General Purpose Framework for Robust Spectroscopic Inference
======================================================================================

We have developed a framework for spectroscopic inference that fulfills the astronomer's dream of using all of the data,
called Starfish. Our statistical framework attempts to overcome many of the difficulties that the astronomer noted.
Principally, at high resolution and high sensitivity, :ref:`model systematics<Fitting Many Lines at Once>`—such as
inaccuracies in the :ref:`strengths of particular lines<Spectral Line Outliers>`—will dominate the noise budget.

We address these problems by accounting for the covariant structure of the residuals that can result from fitting models
to data in this high signal-to-noise, high spectral resolution regime. Using some of the :ref:`machinery<Model the Covariance>` developed by the
field of Gaussian processes, we can parameterize the covariant structure both due to general line mis-matches as well
as specific "outlier" spectral lines due to pathological errors in the atomic and molecular line databases.


**Besides alleviating the problem** of systematic bias and spectral line outliers when :ref:`inferring stellar parameters<Marginalized Stellar Parameters>`,
this approach has many added benefits. By forward-modeling the data spectrum, we put the problem of spectroscopic
inference on true probabilistic footing. Rather than iterating in an open loop between stellar spectroscopists and
stellar modelers, whereby knowledge about the accuracies of line fits is communicated post-mortem, a probabilistic
inference framework like Starfish delivers posterior distributions over the locations and strengths of outlier spectral
lines. Combined with a suite of stellar spectra spanning a range of stellar parameters and a tunable list of atomic and
molecular constants, a probabilistic framework like this provides a way to close the loop on improving both the stellar
models and the stellar parameters inferred from them by comparing models to data directly, rather than mediating through
a series of fits to selected spectral lines.

Lastly, using a forward model means that uncertainties about other non-stellar parameters, such as flux-calibration or
interstellar reddening, can be built into the model and propagated forward. In a future version of Starfish we aim to
include a parameterization for the accretion continuum that "veils" the spectra of young T Tauri stars.

Fitting Many Lines at Once
==========================


Spectral Line Outliers
======================


Model the Covariance
====================


Robust to Outlier Spectral Lines
================================


Marginalized Stellar Parameters
===============================


Spectral Emulator
=================
