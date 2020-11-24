# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os
import re

version = ""

with open(os.path.join("Starfish", "__init__.py"), "r") as fh:
    for line in fh.readlines():
        m = re.search("__version__ = [\"'](.+)[\"']", line)
        if m:
            version = m.group(1)


with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    long_description=readme,
    long_description_content_type="text/markdown",
    name="astrostarfish",
    version=version,
    description="Covariance tools for fitting stellar spectra",
    python_requires="==3.*,>=3.6.0",
    project_urls={
        "repository": "https://github.com/iancze/Starfish",
        "documentation": "https://starfish.rtfd.io",
    },
    author="Ian Czekala",
    author_email="iancze@gmail.com",
    maintainer="Miles Lucas <mdlucas@hawaii.edu",
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
        "astropy==4.*",
        "dataclasses==0.*,>=0.6.0",
        "extinction==0.4.*",
        "flatdict==4.*",
        "h5py==3.*",
        "nptyping==1.*",
        "numpy==1.*,>=1.16.0",
        "scikit-learn==0.*,>=0.21.2",
        "scipy==1.*,>=1.3.0",
        "toml==0.10.*,>=0.10.1",
        "tqdm==4.*",
    ],
    extras_require={
        "docs": [
            "nbsphinx==0.*,>=0.4.2",
            "sphinx==2.3.*",
            "sphinx-bootstrap-theme==0.*,>=0.7.1",
            "IPython",
            "sphinx-autodoc-typehints==1.10.3",
        ],
        "test": [
            "coveralls==1.*,>=1.8.0",
            "pytest==4.*,>=4.6.0",
            "pytest-black",
            "pytest-cov==2.*,>=2.7.0",
            "pytest-benchmark",
        ],
    },
)
