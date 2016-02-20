# Change Log

## [Unreleased]

### Added
- Support for IGRINS and ESPaDOnS instrument classes
- Save intermediate progress in MCMC chains with arg `--incremental_save={N}` in `star.py --sample`
- Worked examples in the Documentation
- Use MCMC sampling optimal jump matrix if it is available with `--use_cov` flag to `star.py --sample`

### Changed
- Minor refactoring

### Fixed
- A bug in how fix_c0 is toggled during Chebyshev polynomial optimization
- The `--cov` feature now works in `chain.py`
- A bug preventing import of model grid interfaces
- Travis builds are now passing
- `sigAmp` is now forced to be positive, preventing Cholesky decomposition error


## [0.1.0] - 2014-01-10
### Added
- First release