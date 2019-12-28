======
Models
======

.. module:: Starfish.models

SpectrumModel
=============

The :class:`SpectrumModel` is the main implementation of the Starfish methods for a single-order spectrum. It works by interfacing with both :class:`Starfish.emulator.Emulator`, :class:`Starfish.spectrum.Spectrum`, and the methods in :mod:`Starfish.transforms`. The spectral emulator provides an interface to spectral model libraries with a covariance matrix for each interpolated spectrum. The transforms provide the physics behind alterations to the light. For a given set of parameters, a transformed spectrum and covariance matrix are provided by

.. code-block:: python

    >>> from Starfish.models import SpectrumModel
    >>> model = SpectrumModel(...)
    >>> flux, cov = model()

It is also possible to optimize our parameters using the interfaces provided in :meth:`SpectrumModel.get_param_vector`, :meth:`SpectrumModel.set_param_vector`, and :meth:`SpectrumModel.log_likelihood`. A very minimal example might be

.. code-block:: python

    >>> from Starfish.models import SpectrumModel
    >>> from scipy.optimize import minimize
    >>> model = SpectrumModel(...)
    >>> def nll(P):
            model.set_param_vector(P)
            lnprob = model.log_likelihood()
            return -lnprob
    >>> P0 = model.get_param_vector()
    >>> soln = minimize(nll, P0, method='Nelder-Mead')

For a more thorough example, see the :doc:`../examples/index`. 

Parametrization
---------------

This model uses a method of specifying parameters very similar to Dan Foreman-Mackey's George library. There exists an underlying dictionary of the model parameters, which define what transformations will be made. For example, if ``vz`` exists in a model's parameter dictionary, then doppler shifting will occur when calling the model. 

It is possible to have a parameter that transforms the spectrum, but is not fittable. We call these `frozen` parameters. For instance, if my 3 model library parameters are :math:`T_{eff}`, :math:`\log g`, and :math:`[Fe/H]` (or ``T``, ``logg``, ``Z`` in the code), but I don't want to fit $\log g$, I can freeze it:

.. code-block:: python

    >>> from Starfish.models import SpectrumModel
    >>> model = SpectrumModel(...)
    >>> model.freeze('logg')

When using this framework, you can see what transformations will occur by looking at :attr:`SpectrumModel.params` and what values are fittable by :meth:`SpectrumModel.get_param_dict` (or the other getters for the parameters). 

.. code-block:: python

    >>> model.params
    {'T': 6020, 'logg': 4.2, 'Z': 0.0, 'vsini': 84.0, 'log_scale': -10.23}
    >>> model.get_param_dict()
    {'T': 6020, 'Z': 0.0, 'vsini': 84.0, 'log_scale': -10.23}

To undo this, simply thaw the frozen parameters

.. code-block:: python

    >>> model.thaw('logg')
    >>> model.params == model.get_param_dict()
    True

The underlying parameter dictionary is a special flat dictionary. Consider the nested dictionary

.. code-block:: python

    >>> cov = {
    ...     'log_amp': 0,
    ...     'log_ls': 6,
    ... }
    >>> model['global_cov'] = cov
    >>> model.params
    {
        'T': 6020, 
        'logg': 4.2, 
        'Z': 0.0, 
        'vsini': 84.0, 
        'log_scale': -10.23,
        'global_cov:log_amp': 0,
        'global_cov:log_ls': 6
    }

These values can be referenced normally or using its flat key

.. code-block:: python

    >>> model['global_cov:log_amp'] == model['global_cov']['log_amp']
    True


API/Reference
-------------

.. autoclass:: SpectrumModel
    :members: 
    :special-members: __call__, __getitem__, __setitem__


Utils
=====

There are some utilities that help with interfacing with the various Models

.. autofunction:: find_residual_peaks
.. autofunction:: optimize_residual_peaks
.. autofunction:: covariance_debugger
