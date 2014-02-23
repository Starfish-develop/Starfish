Grid Tools
==========

====================
Module level methods
====================

.. automodule:: grid_tools
   :members: resample_and_convolve, gauss_taper

==========
Exceptions
==========

.. autoexception:: GridError

.. autoexception:: InterpolationError




===============
Grid Interfaces
===============

.. inheritance-diagram:: RawGridInterface PHOENIXGridInterface KuruczGridInterface BTSettlGridInterface
   :parts: 1

.. automodule:: grid_tools
   :members: RawGridInterface, PHOENIXGridInterface, HDF5Interface

============
Grid Creator
============

.. automodule:: grid_tools
   :members: HDF5GridCreator, MasterToFITSProcessor


=============
Interpolators
=============

.. automodule:: grid_tools
   :members: IndexInterpolator, Interpolator

===========
Instruments
===========

.. inheritance-diagram:: Instrument KPNO TRES Reticon
   :parts: 1

.. automodule:: grid_tools
   :members: Instrument, KPNO, TRES, Reticon



