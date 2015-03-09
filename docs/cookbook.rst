========
Cookbook
========

Download the spectral library to a good location on your disk.


Create a local working directory for the star you wish to process.

Copy `config.yaml` to this directory and modify the settings as you wish. It is generally a good idea to keep the HDF5 paths set to this local directory.

From within this local directory, we will want to create the grid.

    # Downsample the grid
    grid.py create

    # Plot all of the spectra in the hdf5 grid into the plotdir
    grid.py plot


Here we should also have some routines to plot the various spectra within the grid to make sure everything looks as we want.

Then, we will want to make the PCA grid, and do many other things regarding optimization and final plotting. These are all done through the `pca.py` script.

    # create the grid using settings in config.yaml
    pca.py --create

    # Plot all of the eigenspectra and the histogram of weights
    pca.py --plot=eigenspectra

    # reconstruct the grid at the synthetic grid points, plotting the difference
    pca.py --plot=reconstruct

    # Specify and examine priors stored in config.yaml
    pca.py --plot=priors

    # Optimize the emulator using fmin
    pca.py --optimize=fmin

    # Optimize the emulator starting from fresh parameter combinations
    pca.py --optimize=fmin --resume

    # OR optimize using emcee
    # default samples = 100
    pca.py --optimize=emcee --samples=100
    pca.py --optimize=emcee --resume

    # Make a triangle plot of the emcee output
    pca.py --plot=emcee

    # Make some plots showing weight interpolations using the emulator
    pca.py --plot=emulator --params=fmin
    # OR
    pca.py --plot=emulator --params=emcee

    # Once you've OK'd the parameters, then store them to the HDF5 file
    pca.py --store --params=fmin
    # OR
    pca.py --store --params=emcee


After doing some more analysis, we'll want to make a bunch of plots showing the scatter of interpolated spectra against what the gridpoint spectrum looks like.

Now that you've optimized the emulator for the specific spectrum you'd like to fit, we can use a series of tools to fit the spectrum. Further customization will require writing your own python scripts.

These codes are much simpler and just output everything to the local directory.

Optimize the grid and observational parameters (:math:`\Theta`)

    star.py --optimize=Theta
    star.py --sample=Theta

Optimize the noise parameters (:math:`\Phi`)

    star.py --optimize=Phi

    star.py --optimize=Phi --regions

Using the parallel code, sample both :math:`\Theta` and :math:`\Phi` at the same time. This code is more robust and actually sets up directories for output and everything.

    pstar.py
