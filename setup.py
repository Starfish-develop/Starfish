import sys
from setuptools import setup, Extension
import builtins

with open("README.md", "r") as fh:
    long_description = fh.read()

try:
    from Cython.Distutils import build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext

    ext_modules = [Extension("Starfish.covariance",
                             sources=["Starfish/covariance.c"],
                             extra_compile_args=["-Wno-declaration-after-statement",
                                                 "-Wno-error=declaration-after-statement",
                                                 "-Wno-unused-function",
                                                 "-Wno-unused-variable",
                                                 "-Wno-unused-but-set-variable"]), ]
else:
    ext_modules = [Extension("Starfish.covariance",
                             sources=["Starfish/covariance.pyx"],
                             extra_compile_args=["-Wno-declaration-after-statement",
                                                 "-Wno-error=declaration-after-statement",
                                                 "-Wno-unused-function",
                                                 "-Wno-unused-variable",
                                                 "-Wno-unused-but-set-variable"]), ]

if sys.version < '3.5.2':
    raise RuntimeError('Error: Python 3.6 or greater required for Starfish (using {})'.format(sys.version))

# This is a bit hackish: we are setting a global variable so that the main
# Starfish.__init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being used.
builtins.__STARFISH_SETUP__ = True
import Starfish


# Use this custom class to be able to force numpy installation before using it.
class CustomBuildExt(build_ext):
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        return super().run()



setup(
    name="astrostarfish",
    version=Starfish.__version__,
    author="Ian Czekala",
    author_email="iancze",
    packages=["Starfish"],
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
        "Topic :: Scientific/Engineering :: Physics"
    ],
    install_requires=['numpy',
                      'scipy',
                      'extinction'
                      'cython',
                      'scikit-learn',
                      'emcee',
                      'h5py',
                      'corner',
                      'astropy',
                      'tqdm',
                      'oyaml'],
    maintainer="Ian Czekala",
    maintainer_email="iancze@gmail.com",
    cmdclass={'build_ext': CustomBuildExt},
    ext_modules=ext_modules,
    include_package_data=True,
)
