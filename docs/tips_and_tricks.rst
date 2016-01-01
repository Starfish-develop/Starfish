==========================
Tips and Tricks
==========================


There are a few tips and tricks, mostly having to do with making cool plots.


Time how long it takes your script to run.
===========================================

Just calling `time` before your script will output the running time:

.. code-block:: bash

	$ time echo $SHELL
	/bin/bash

	real	0m0.000s
	user	0m0.000s
	sys	0m0.000s


The output is interesting.  It provides the `real` (wall-clock) time it took to execute.  But it also provides the `user` and `sys` times.  `user` is generally larger than `real`, due to parallel processing.  `sys` is usually less than `real` becayse your computer is doing other things other than just the shell task.  But I (gully) have seen strange behavior with `time` that challenges this simplistic view.  Tips here are welcome.


Tuning the MCMC jump sizes
============================
There's some subtlety to this that I (gully) do not fully understand.  See Section 4 of Foreman-Mackey et al. (2013) for a discussion and references.  Luckily, there are some good strategies for picking "optimal" jumps.  Try this:

.. code-block:: bash

	$ chain.py --files mc.hdf5 --cov

	[...]

	'Optimal' jumps
	[  1.65734629e+01   6.46752715e-02   2.66108819e-02   4.29205617e-02
	   1.18523979e-01   5.78843022e-04]

Now you can just type these into the *Theta_jump* section of your `config.yaml` file.  You'll probably want to update the jumps for your 


How many CPUs do you have.
============================

There are a few ways to do this.  In IPython:

.. code-block:: python

	In [1]: import multiprocessing as mp

	In [2]: mp.cpu_count()
	Out[2]: 8


In bash:

.. code-block:: bash

	$ getconf _NPROCESSORS_ONLN
	8
