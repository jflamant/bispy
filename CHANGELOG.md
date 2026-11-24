# Changelog

All notable changes to BiSPy are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow
[PEP 440](https://peps.python.org/pep-0440/).

Releases up to 0.9.4.dev0 were development snapshots and have no changelog.

## [Unreleased]

### Changed
- BiSPy is now distributed under the MIT License (previously CC-BY-4.0, and CeCILL before 2023).
- Python 3.10 or later is required.
- Packaging moved from `setup.py` to `pyproject.toml`, built with uv. The PyPI name
  (`bispy-polar`) and the import name (`bispy`) are unchanged.

### Added
- `bispy.__version__`.
- A test suite covering all modules, run in CI on Linux, macOS and Windows.

### Fixed
- `quaternionPSD.plot()` and `quaternionPSD.plotStokes()` crashed with matplotlib 3.1 and later.
- `quaternionPSD.plot(single_sided=False)` limited the frequency axis to (-π/4, π/4).
- Plotting functions advanced NumPy's global random number generator (a random array set
  the figure aspect ratio), so making a plot changed the random numbers drawn afterwards.
- `utils.visual.plot2D` now accepts complex input, as documented.
- `spectral.quaternionPSD`, `filters.HermitianFilter` and `filters.UnitaryFilter` (and therefore
  `signals.stationaryBivariate`) failed for odd signal lengths.
- `HermitianFilter` and `UnitaryFilter` overwrote the caller's `K`, `eta`, `alpha` and `phi`
  arrays with their symmetrized versions. The symmetrized arrays remain available as the
  filter attributes `K`, `eta`, `alpha` and `phi`.
- Adding two spectral estimates (`Periodogram`, `Multitaper`) computed on different time
  arrays was silently accepted. It now raises `ValueError`, as intended.
- `Multitaper` `+` and `*` returned an object with the default bandwidth, whose per-taper
  `densities` and `dpss` did not match its `density`. Results now keep the bandwidth and the
  combined per-taper densities. Adding two `Multitaper` estimates with different bandwidths
  raises `ValueError`.
- With `computeFlag=False`, a later call to `compute()` did not update the Stokes parameters
  (nor, for `Multitaper`, `density`), and `Multitaper.compute()` ignored the bandwidth given to
  the constructor.

[Unreleased]: https://github.com/jflamant/bispy/compare/v0.9.4.dev...HEAD
