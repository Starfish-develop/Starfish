=================================
Example: WASP 14 Multiple Orders
=================================


This is an example of running Starfish on multiple spectral orders with shared stellar properties.  I will mainly focus on the parts of the example that are different from Example 1.  This example assumes you have already performed Example 1 a-c.


Getting set up
=====================
The first step is to copy the configuration and parameter files from Example 1 (`demo1/`) into this directory (`demo2/`).  We want to use the output parameter values from the previous run as the guesses for this run.

Copy from demo1 to demo2:

1. `config.yaml`
2. `data/`
3. The directory structure

Then we modify our configuration file to the desired orders and wavelengths:

.. code-block:: yaml
    :emphasize-lines: 3, 9, 22, 23

    # YAML configuration script

        name: wasp14_trial1002

        data:
          grid_name: "PHOENIX"
          files: ["data/WASP14/WASP14-2009-06-14.hdf5"]
          instruments : ["TRES"]
          orders: [21, 22, 23, 24]

        outdir : output/

        plotdir : plots/

        # The parameters defining your raw spectral library live here.
        grid:
          raw_path: "/Users/gully/GitHub/Starfish/libraries/raw/PHOENIX/"
          hdf5_path: "libraries/PHOENIX_TRES_test.hdf5"
          parname: ["temp", "logg", "Z"]
          key_name: "t{0:.0f}g{1:.1f}z{2:.1f}" # Specifies how the params are stored
          # in the HDF5 file
          parrange: [[6000, 6400], [4.0, 5.0], [-1.0, 0.0]]
          wl_range: [4975, 5415]
          buffer: 50. # AA


In this example I will optimize the spectral emulator with `emcee` MCMC sampling rather than the `fmin` optimization.  The main virtue of emcee, from a practical standpoint, is that it does not require user intervention (`fmin` usually hits a 10,000 max-iterations limit, in my experience).  So I ran `pca.py --optimize=emcee --samples=5000`, which took about 6 hours on my Macbook Pro.  This was probably overkill.  Recall that this process is simply optimizing the hyperparameters for the amplitude and scale of the Gaussian Process fit of the eigenspectra weights.  Here is an example image:

.. image:: assets/triangle_0.png


# Only spot check the last order
==================================

Only spot check the last order.  Why?  Because this is the order in which the :math:`c^{0}` term is fixed.  So adjust **logOmega** in the `config.yaml` file until the continuum levels match up.  The continuum mismatches in the other orders will be taken up by the non-zero :math:`c^{0}` term in :math:`\phi{P}`.  I suspect it is faster to do this by eye, than letting the computer optimize for you.  I like to make a tweak and then run:

.. code-block:: bash

    $ star.py --generate; splot.py s0_o24spec.json --matplotlib; open plots/s0_o24spec.json.png

I also hand-adjusted my `theta.json` file, since the theta optimization was taking a long time.  The `optimize=Cheb` command works fine because the problem structure is natively parallelized.  

# MCMC
========
Get your run01 directory set up, and edit the `config.yaml` as we did in the previous example. 

.. code-block:: bash

    $ ls
    config.yaml    s0_o21         s0_o22         s0_o23         s0_o24
    plots          s0_o21phi.json s0_o22phi.json s0_o23phi.json s0_o24phi.json

Five hundred ThetaCheb samples took 36 minutes on my Macbook pro.

.. code-block:: bash

    $ time star.py --sample=ThetaCheb --samples=500
    Final [  6.21445629e+03   4.21662234e+00  -2.70602364e-01  -4.79946024e+00
       5.80021176e+00  -1.27221067e+01]

    real  36m12.803s
    user  90m17.047s
    sys 12m43.253s

As usual, hand-edit your `config.yaml` file and `phi.json` files with your best guess for revised stellar and calibration parameters.  Then you can introduce yet-more-parameters.  It took my computer 40 minutes to run 500 samples with all parameters:

.. code-block:: bash

    $ time star.py --sample=ThetaPhi --samples=500

    Final [  6.38428451e+03   4.18315434e+00  -3.21968427e-01  -4.88951948e+00
       5.33927051e+00  -1.27197478e+01]

    real  40m17.345s
    user  106m42.716s
    sys 14m27.443s

