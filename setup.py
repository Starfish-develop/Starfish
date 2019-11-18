# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import builtins

builtins.__STARFISH_SETUP__ = True

import Starfish

with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    long_description=readme,
    long_description_content_type="text/markdown",
    name="astrostarfish",
    version=Starfish.__version__,
    description="Covariance tools for fitting stellar spectra",
    python_requires="==3.*,>=3.6.0",
    project_urls={
        "repository": "https://github.com/iancze/Starfish",
        "documentation": "https://starfish.rtfd.io",
    },
    author="Ian Czekala",
    author_email="iancze@gmail.com",
    maintainer="Miles Lucas <mdlucas@iastate.edu",
    license="BSD-4-Clause",
    keywords="Science Astronomy Physics Data Science",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_data={},
    install_requires=[
        "astropy==3.*,>=3.1.0",
        "dataclasses==0.*,>=0.6.0",
        "emcee==3.0",
        "extinction @ git+https://github.com/kbarbary/extinction.git@master#egg=extinction",
        "flatdict==3.*,>=3.3.0",
        "h5py==2.*,>=2.9.0",
        "nptyping==0.*,>=0.2.0",
        "numpy==1.*,>=1.16.0",
        "scikit-learn==0.*,>=0.21.2",
        "scipy==1.*,>=1.3.0",
        "toml @ git+https://github.com/uiri/toml.git@master#egg=toml-0.10.1",
        "tqdm==4.*,>=4.32.0",
    ],
    extras_require={
        "docs": [
            "nbsphinx==0.*,>=0.4.2",
            "sphinx==2.*,>=2.1.0",
            "sphinx-bootstrap-theme==0.*,>=0.7.1",
            "IPython",
            "sphinx-autodoc-typehints",
        ],
        "test": [
            "coveralls==1.*,>=1.8.0",
            "pytest==4.*,>=4.6.0",
            "pytest-cov==2.*,>=2.7.0",
            "pytest-benchmark",
        ],
    },
)
