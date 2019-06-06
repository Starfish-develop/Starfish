import Starfish
from setuptools import setup, find_packages
import builtins

with open("README.md", "r") as fh:
    long_description = fh.read()

# This is a bit hackish: we are setting a global variable so that the main
# Starfish.__init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being used.
builtins.__STARFISH_SETUP__ = True


setup(
    name="astrostarfish",
    version=Starfish.__version__,
    python_requires=">=3.6.0",
    author="Ian Czekala",
    author_email="iczekala@gmail.com",
    packages=find_packages(),
    url="https://github.com/iancze/Starfish",
    download_url="https://github.com/iancze/Starfish/archive/master.zip",
    license="BSD",
    description="Covariance tools for fitting stellar spectra",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=[
        "astropy",
        "emcee==3.0rc2",
        "extinction",
        "h5py",
        "nptyping",
        "numpy",
        "scikit-learn",
        "scipy",
        "toml",
        "tqdm",
    ],
    # Need this link to get the most up-to-date toml version
    dependency_links=["https://github.com/uiri/toml/tarball/master#egg=toml"],
    maintainer="Ian Czekala",
    maintainer_email="iancze@gmail.com",
    include_package_data=True,
    test_requires=["pytest", "pytest-cov", "pytest-black", "mypy"],
)
