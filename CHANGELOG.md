# Changelog

All notable changes to BiSPy are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Since 1.0.0, versions follow
[semantic versioning](https://semver.org/): changes that break existing code only happen
in a new major version.

Releases up to 0.9.4.dev0 were development snapshots and have no changelog.

## [Unreleased]

### Added
- `utils.polarizationAxis(theta, chi)` returns the polarization axis of an ellipse: the
  unit pure quaternion i S3 + j S1 + k S2 (normalized Stokes parameters), as expected by
  `spectral.quaternionPSD` and the filters. The filtering tutorial uses it.
- The spectral analysis tutorial explains why estimates are averaged: the periodogram of a
  single realization always has a degree of polarization of 1. Adapted from the second
  notebook of the EUSIPCO 2024 tutorial.
- `timefrequency.Hembedding.plot()` displays the instantaneous amplitude, orientation,
  ellipticity and phase. The phase is unwrapped, including the jumps of pi that occur
  when the orientation wraps around.
- The time-frequency tutorial starts with a section on the quaternion embedding and the
  instantaneous polarization parameters, adapted from the third notebook of the EUSIPCO
  2024 tutorial.
- `CITATION.cff` gives the Zenodo concept DOI of BiSPy, so GitHub's "Cite this repository"
  button includes it.
- A getting-started tutorial: bivariate signals as quaternion arrays, the polarization
  ellipse and its quaternion polar form, Stokes parameters and the Poincaré sphere, and
  the QFT. It is adapted from the first notebook of the EUSIPCO 2024 tutorial.
- A tutorial on linear filtering (`filters` module): simulating a signal with a prescribed
  quaternion PSD with a Hermitian filter, rotating polarization states with a unitary
  filter, and splitting a signal in two with a Hermitian filter (an unpolarized remainder,
  or two uncorrelated polarized parts). Parts of it are adapted from the EUSIPCO 2024
  tutorial by N. Le Bihan and J. Flamant.

### Changed
- The docstring of `timefrequency.Hembedding` states the expected input: the bivariate
  signal itself, not its quaternion embedding.
- The documentation uses the Sphinx Book Theme instead of the Read the Docs theme. Each
  page links to the GitHub repository, to the issue tracker and to its own source for edits.

### Fixed
- `utils.Stokes2geo` and `utils.normalizeStokes` no longer emit `RuntimeWarning`s where
  there is no power (S0 = 0) or no polarization (Phi = 0). They return NaN there, as
  before. So `spectral.quaternionPSD.plot()` is now silent for band-limited PSDs.
- `utils.normalizeStokes`, and so `utils.Stokes2geo`, ignore NaN values in S0 when
  computing the tolerance. Before, a single NaN made every output NaN, even with `tol=0`.
- Math written as `$...$` in the tutorials is rendered as equations, instead of being
  shown as raw LaTeX.

## [1.0.0] - 2026-11-25

First stable release, archived on Zenodo. The code is the same as in 0.10.0, apart from
documentation fixes.

### Added
- `CITATION.cff`: GitHub's "Cite this repository" button and Zenodo use it to cite BiSPy.
- The documentation includes this changelog and documents `utils.polarizationEllipse`.

### Changed
- The tutorials are Jupyter notebooks run every time the documentation is built, so their
  outputs always match the current code. The notebooks moved from `notebooks/` to
  `docs/tutorials/`.

### Fixed
- The formula in the `utils.StokesNorm` docstring had a wrong sign (the code was correct).

## [0.10.0] - 2026-11-25

First release after the 0.9.x development snapshots. When upgrading, note that:
- results change for `QCWT` with the Morlet wavelet (`S0` to `S3` are 4 times larger with the
  default parameters) and for `Hembedding` of odd-length signals, which were wrong before;
- progress messages are no longer printed but logged (see Changed);
- some invalid operations now raise `ValueError` instead of silently giving wrong results.

### Changed
- BiSPy is now distributed under the MIT License (previously CC-BY-4.0, and CeCILL before 2023).
- Python 3.10 or later is required.
- Packaging moved from `setup.py` to `pyproject.toml`, built with uv. The PyPI name
  (`bispy-polar`) and the import name (`bispy`) are unchanged.
- Progress messages ("Computing Time-Frequency Stokes parameters", "Ridge added",
  "Number of data tapers: ...") are no longer printed. They go through Python's `logging`
  module (loggers `bispy.timefrequency` and `bispy.spectral`) and are silent unless enabled,
  for example with `logging.basicConfig(level=logging.INFO)`.
- `utils.visual.plot3D`, `Periodogram.plot` and `Multitaper.plot` no longer call
  `fig.show()`. Figures still display in notebooks; scripts call `plt.show()` as usual.

### Added
- `bispy.__version__`.
- A test suite covering all modules, run in CI on Linux, macOS and Windows.
- `rng` argument for `signals.bivariatewhiteNoise` and `signals.stationaryBivariate.simulate`.
  A seed or a `numpy.random.Generator` gives reproducible results without touching NumPy's
  global random state. The default, `rng=None`, keeps using the global state, so results
  seeded with `np.random.seed` are unchanged.
- `Hembedding` option of `signals.bivariatewhiteNoise`, returning the H-embedding (quaternion
  analytic signal) of the noise ([#8](https://github.com/jflamant/bispy/issues/8)).

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
- `QSTFT.extractRidges()` (and `QSTFT.compute(ridges=True)`) failed when `nfft` was not given.
- `timefrequency.Hembedding` was inexact for odd signal lengths: the highest positive frequency
  was halved, so the embedding no longer contained the original signal. Even lengths are
  unchanged.
- `QSTFT.plotRidges()` and `QCWT.plotRidges()` crashed when no ridge was found.
- An unknown `norm` option for Morse wavelets in `QCWT.compute` raised an obscure
  `UnboundLocalError`. It now raises a `ValueError` naming the valid options, `'bandpass'`
  and `'energy'`.
- **Values change:** `QCWT` with the Morlet wavelet used a wrong normalization constant
  (`(...)**1/2` instead of `(...)**(-1/2)`). Coefficients were too small by a factor
  2·sqrt(1 + exp(-η²) - 2 exp(-3η²/4)), which equals 2 for the default η = 2π. With the default,
  `tfpr` is now 2 times larger and `S0`, `S1`, `S2`, `S3` are 4 times larger. Normalized Stokes
  parameters, orientation, ellipticity and ridges are unchanged. Morse wavelets are not affected.

[Unreleased]: https://github.com/jflamant/bispy/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/jflamant/bispy/compare/v0.10.0...v1.0.0
[0.10.0]: https://github.com/jflamant/bispy/compare/v0.9.4.dev...v0.10.0
