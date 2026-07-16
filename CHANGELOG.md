# Change Log

## [Unreleased]

### Added
- Support for IGRINS and ESPaDOnS instrument classes
- Save intermediate progress in MCMC chains with arg `--incremental_save={N}` in `star.py --sample`
- Worked examples in the Documentation
- Use MCMC sampling optimal jump matrix if it is available with `--use_cov` flag to `star.py --sample`

### Changed
- Minor refactoring
- Support Python 3.11 through 3.14; dropped support for Python 3.10 and earlier
- Packaging metadata moved from `setup.py` to PEP 621 `pyproject.toml`
- Removed the `nptyping` dependency in favor of `numpy.typing`, and the obsolete
  `dataclasses` backport; relaxed upper version caps on `astropy`, `numpy`,
  `scipy`, `scikit-learn`, `extinction`, `h5py`, `flatdict`, `toml`, and `tqdm`
- Replaced `np.trapz` with `scipy.integrate.trapezoid` and `np.Inf` with
  `np.inf` for NumPy 2 compatibility

### Fixed
- A bug in how fix_c0 is toggled during Chebyshev polynomial optimization
- The `--cov` feature now works in `chain.py`
- A bug preventing import of model grid interfaces
- Travis builds are now passing
- `sigAmp` is now forced to be positive, preventing Cholesky decomposition error


## [0.1.0] - 2014-01-10
### Added
- First release