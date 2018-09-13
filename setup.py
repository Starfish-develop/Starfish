#!python

import sys
if sys.version < '3.3':
    sys.exit('Error: Python 3.3 or greater required for Starfish (using {})'.format(sys.version))

import ez_setup
ez_setup.use_setuptools()

from setuptools import setup, Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    print("Please install Cython, using either \n" +
          "$sudo pip install cython\n" +
          "OR http://docs.cython.org/src/quickstart/install.html")

try:
    import numpy as np
except ImportError:
    print("Please install numpy: http://www.scipy.org/install.html")


LIBRARY_DIRS = []
INCLUDE_DIRS = [np.get_include()]# + ["YOUR_INCLUDE_DIR_HERE"]
#LIBRARY_DIRS = ["YOUR_LIB_DIR_HERE"]


if __name__=="__main__":
    import argparse
    #Pass LIBRARY DIRS and INCLUDE_DIRS as command line arguments
    parser = argparse.ArgumentParser(prog="setup.py", description="Setup file for Starfish.")
    parser.add_argument("-L", "--L", help="Location of SuiteSparse C libraries, if a custom directory chosen.")
    parser.add_argument("-I", "--I", help="Location of SuiteSparse C headers, if a custom directory chosen.")
    args, unknown = parser.parse_known_args() #handy trick from dfm/george
    sys.argv = [sys.argv[0]] + unknown

    if args.L: #lib is not None
        LIBRARY_DIRS += [args.L]

    if args.I: #include is not None
        INCLUDE_DIRS += [args.I]

    setup(
            name="Starfish",
            version="0.1",
            author="Ian Czekala",
            author_email="iancze",
            packages=["Starfish"],
            url="https://github.com/iancze/Starfish",
            download_url="https://github.com/iancze/Starfish/archive/master.zip",
            license="BSD",
            description="Covariance tools for fitting stellar spectra",
            classifiers=[
                "Intended Audience :: Science/Research",
                "Programming Language :: Python :: 3",
                "Topic :: Scientific/Engineering :: Astronomy",
                "Topic :: Scientific/Engineering :: Physics"
            ],
            install_requires = ['numpy', 'scipy', 'cython', 'scikit-learn', 'emcee', 'h5py', 'corner'],
            maintainer = "Ian Czekala",
            maintainer_email = "iancze@gmail.com",
            cmdclass = {'build_ext' :build_ext},
            ext_modules = [Extension("Starfish.covariance",
                ["Starfish/covariance.pyx"],
                #libraries=['m', 'cholmod', 'amd', 'colamd', 'blas', 'lapack', 'suitesparseconfig', 'rt'],
                include_dirs=[np.get_include()],
                #include_dirs=INCLUDE_DIRS,
                #library_dirs=LIBRARY_DIRS,
                extra_compile_args=["-Wno-declaration-after-statement",
                                    "-Wno-error=declaration-after-statement",
                                    "-Wno-unused-function",
                                    "-Wno-unused-variable",
                                    "-Wno-unused-but-set-variable"])]
            #extra_compile_args are necessary to compile extern/cov.h in python 3.4
    )

#By looking at /usr/lib/python3.4/distutils, we are able to dig deeper into the problem at get_config_vars around line 188
#It seems as though any command line option does not override the flags used by distutils.

#Previously, this was needed at the top of setup.py
#import os
#os.environ["OPT"] = "-DDYNAMIC_ANNOTATIONS_ENABLED=1 -DNDEBUG -g -fwrapv -O3"
#os.environ["CFLAGS"] = "-march=x86-64 -mtune=generic -O2 -pipe -fstack-protector --param=ssp-buffer-size=4"

#Now I compile with
#OPT="-DDYNAMIC_ANNOTATIONS_ENABLED=1 -DNDEBUG -g -fwrapv -O3" CFLAGS="-march=x86-64 -mtune=generic -O2 -pipe -fstack-protector --param=ssp-buffer-size=4" python setup.py build_ext --inplace

# Installation of SuiteSparse

#    python setup.py build_ext --inplace -Lmydir/lib -Imydir/include

#If you want to try using my SuiteSparse directories on the CfA CF network,

#python setup.py build_ext --inplace -L/pool/scout0/.build/lib -I/pool/scout0/.build/include
