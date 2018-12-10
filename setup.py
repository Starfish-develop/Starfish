import sys
from setuptools import setup, Extension
import builtins

with open("README.md", "r") as fh:
    long_description = fh.read()

try:
    from Cython.Distutils import build_ext
except ImportError:
    print("Please install Cython, using either \n" +
          "$sudo pip install cython\n" +
          "OR http://docs.cython.org/src/quickstart/install.html")


if sys.version < '3.3':
    raise RuntimeError('Error: Python 3.3 or greater required for Starfish (using {})'.format(sys.version))

# This is a bit hackish: we are setting a global variable so that the main
# Starfish.__init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being used.
builtins.__STARFISH_SETUP__ = True
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
        install_requires = ['numpy',
                            'scipy',
                            'cython',
                            'scikit-learn',
                            'emcee==3.0rc2',
                            'h5py',
                            'corner',
                            'astropy',
                            'tqdm',
                            'pyyaml'],
        maintainer = "Ian Czekala",
        maintainer_email = "iancze@gmail.com",
        cmdclass = {'build_ext' :build_ext},
        ext_modules = [Extension("Starfish.covariance",
            ["Starfish/covariance.pyx"],
            include_dirs=['extern'],
            extra_compile_args=["-Wno-declaration-after-statement",
                                "-Wno-error=declaration-after-statement",
                                "-Wno-unused-function",
                                "-Wno-unused-variable",
                                "-Wno-unused-but-set-variable"])],
        long_description = long_description,
        long_description_content_type = "text/markdown",
)
