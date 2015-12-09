==========================
Troubleshooting
==========================


Here we will cover some possible scenarios that could cause you to get tripped up.


Problems generating model: Bad wavelength range
================================================

If you use the default `config.yaml` file that comes with Starfish, you might encounter a problem like this:


.. code-block:: bash

    $ star.py --generate

    [...]
    AssertionError: determine_chunk_log: wl_min 5084.98 and wl_max
       5285.93 are not within the bounds of the grid 4950.00 to 5250.00.
    [...]


The origin of this problem should be evident from the error message- your wavelength bounds in your model and data do not match.  This makes sense: you simply told the code the wavelength range in the config.yaml file.  The code then fetched the instrumental resolution from the instrument properties.  When you go to compare the data that is outside that range, the code will fail.  A common problem is being "off-by-one" in the order number(s): did you zero- or one- index the order numbers?  Inspect the order of interest with `HDFView.app` to make sure you have the right boundaries, and consider increasing the wavelength padding. 


NoneType and parameters out of range
================================================

If you use the default `config.yaml` file that comes with Starfish, you might encounter a problem like this:


.. code-block:: bash

    $ star.py --generate
    keeping grid as is
    grid pars are [  5.99900000e+03   4.22000000e+00  -2.60000000e-01]
    Process Process-1:
    Traceback (most recent call last):
      File "/anaconda/lib/python3.4/multiprocessing/process.py", line 254, in _bootstrap
        self.run()
      File "/anaconda/lib/python3.4/multiprocessing/process.py", line 93, in run
        self._target(*self._args, **self._kwargs)
      File "/Users/gully/GitHub/Starfish/Starfish/parallel.py", line 533, in brain
        alive = self.interpret()
      File "/Users/gully/GitHub/Starfish/Starfish/parallel.py", line 551, in interpret
        response = func(arg)
      File "/Users/gully/GitHub/Starfish/Starfish/parallel.py", line 585, in save
        model = self.chebyshevSpectrum.k * self.flux_mean + X.dot(self.mus)
    TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'


This cryptic error message had me confused for quite a while.  It traces back to the fact that `self.mus` is a `NoneType`, which led me down a rabbit hole to figure out why.  It turns out the solution to my particular problem was quite simple.  To understand, inspect the config.yaml file:

.. code-block:: yaml
    :emphasize-lines: 25, 40

    # YAML configuration script

    name: example_wasp14

    data:
      grid_name: "PHOENIX"
      files: ["data/WASP14/WASP14-2009-06-14.hdf5"]
      # data/WASP14/WASP14-2010-03-29.hdf5
      # data/WASP14/WASP14-2010-04-24.hdf5
      instruments : ["TRES"]
      orders: [21]
      #orders: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

    outdir : output/

    plotdir : plots/

    # The parameters defining your raw spectral library live here.
    grid:
      raw_path: "/Users/gully/GitHub/Starfish/libraries/raw/PHOENIX/"
      hdf5_path: "libraries/PHOENIX_TRES_test.hdf5"
      parname: ["temp", "logg", "Z"]
      key_name: "t{0:.0f}g{1:.1f}z{2:.1f}" # Specifies how the params are stored
      # in the HDF5 file
      parrange: [[6000, 6300], [4.0, 5.0], [-1.0, 0.0]]
      wl_range: [5000, 5200]
      buffer: 50. # AA

    PCA:
      path : "PHOENIX_TRES_PCA.hdf5"
      threshold: 0.999 # Percentage of variance explained by components.
      priors: [[2., 0.0075], [2., 0.75], [2., 0.75]] # len(parname) list of 2-element lists. Each 2-element list is [s, r] for the Gamma-function prior on emulator parameters

    #Longer strings can be written like this. This will be loaded under the "Comments" variable.
    Comments: >
      WASP14 spectrum using emulator.

    # The parameters shared between all orders
    Theta :
        grid : [5999., 4.22, -0.26]
        vz : -4.77
        vsini : 5.79
        logOmega: -12.80
        Av: 0.0

In this case, my guess for the effective temperature value of 5999 K was less than the 6000 K lower boundary of the grid.  If your **Theta:** parameters are outside of the bounds of your **parrange** ranges, it causes the error message above, but with no indication what is wrong.  We should probably add some sort of checking for this as soon as the `config.yaml` file is read-in, but hopefully users will avoid making the same mistake I did!


`fix_c0` is incorrectly disabled 
==================================
`fix_c0` is incorrectly disabled when star.py --optimize=Cheb is run on a single order.

