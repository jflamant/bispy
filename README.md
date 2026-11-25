# BiSPy : Bivariate Signal Processing with Python

[![docs-page](https://img.shields.io/badge/docs-latest-blue)](https://jflamant.github.io/bispy/)
[![PyPI version](https://badge.fury.io/py/bispy-polar.svg)](https://badge.fury.io/py/bispy-polar)
[![Tests](https://github.com/jflamant/bispy/actions/workflows/tests.yml/badge.svg)](https://github.com/jflamant/bispy/actions/workflows/tests.yml)

BiSPy is an open-source python framework for processing bivariate signals. It supports our papers on time-frequency analysis [1], spectral analysis [2] and linear time-invariant filtering [3] of bivariate signals.

> [1] Julien Flamant, Nicolas Le Bihan, Pierre Chainais: “Time-frequency analysis of bivariate signals”, Applied and Computational Harmonic Analysis, 2017; [arXiv:1609.02463](http://arxiv.org/abs/1609.02463), [doi:10.1016/j.acha.2017.05.007](https://doi.org/10.1016/j.acha.2017.05.007)

> [2] Julien Flamant, Nicolas Le Bihan, Pierre Chainais: “Spectral analysis of stationary random bivariate signals”, 2017, IEEE Transactions on Signal Processing; [arXiv:1703.06417](http://arxiv.org/abs/1703.06417), [doi:10.1109/TSP.2017.2736494](https://doi.org/10.1109/TSP.2017.2736494)

> [3] Julien Flamant, Pierre Chainais, Nicolas Le Bihan: “A complete framework for linear filtering of bivariate signals”, 2018; IEEE Transactions on Signal Processing; [arXiv:1802.02469](https://arxiv.org/abs/1802.02469), [doi:10.1109/TSP.2018.2855659](https://doi.org/10.1109/TSP.2018.2855659)

These papers contain theoretical results and several applications that can be reproduced with this toolbox.


BiSPy is developed on GitHub. If you encounter a bug or something unexpected, please let us know by [raising an issue](https://github.com/jflamant/bispy/issues) on the project page.

## Install

Because of a name conflict, the package is published on PyPI as `bispy-polar`. To install it, type

```
pip install bispy-polar
```

or, in a [uv](https://docs.astral.sh/uv/) project, `uv add bispy-polar`. Dependencies are installed automatically (see below).

To get started, simply use 
```
import bispy as bsp
```

## Requirements

BiSPy works with Python 3.10+.

Dependencies:
 -   [NumPy](http://www.numpy.org)
 -   [SciPy](https://www.scipy.org)
 -   [Matplotlib](http://matplotlib.org)
 -   [numpy-quaternion](https://github.com/moble/quaternion)

[numpy-quaternion](https://github.com/moble/quaternion) adds quaternion dtype support to numpy. Implementation by [moble](https://github.com/moble). Since this python toolbox relies extensively on this module, you can check out first the nice introduction [here](https://github.com/moble).


## License

This software is distributed under the [MIT License](https://github.com/jflamant/bispy/blob/master/LICENCE).

## Cite this work

If you use BiSPy in your research, please cite the software, together with the paper(s) describing the methods you use ([1]–[3] above).

To cite the software, use the **"Cite this repository"** button on the [GitHub page](https://github.com/jflamant/bispy), which gives APA and BibTeX entries. They are generated from [`CITATION.cff`](https://github.com/jflamant/bispy/blob/master/CITATION.cff).

## Development

The project is managed with [uv](https://docs.astral.sh/uv/). The file `uv.lock` pins the exact versions of every dependency used for development and CI.

```shell
git clone https://github.com/jflamant/bispy.git && cd bispy
uv sync                      # create .venv with bispy and all dev tools
uv run pytest                # run the test suite
uv run ruff check            # lint
uv run sphinx-build -b html docs docs/_build/html   # build the docs (runs the tutorials)
```

The tutorials are Jupyter notebooks in `docs/tutorials/`, stored without outputs. They run every time the documentation is built, so a tutorial that breaks fails the docs build.

The test suite contains three kinds of tests:
- **Property tests** check mathematical identities (round trips, Parseval, symmetries).
- **Snapshot tests** compare outputs with reference values stored in `tests/<module>/*.npz`. After an intentional change to numerical results, regenerate them with `uv run pytest --force-regen`, check that the change is the expected one, and commit the updated `.npz` files.
- **Known-bug tests** are marked `xfail`; `uv run pytest -rx` lists them. Once a bug is fixed its test passes, and pytest reports a failure until the `xfail` marker is removed.

To make a release:
1. Set the version with `uv version X.Y.Z`. This updates `pyproject.toml` and `uv.lock`.
2. Date the release in `CHANGELOG.md`, and check that `CITATION.cff` is up to date (the release workflow refuses placeholders marked to-do).
3. Commit, then tag `vX.Y.Z`.
4. Publish a GitHub Release from that tag. CI tests the package, builds it and uploads it to PyPI, and Zenodo archives the release with a DOI.
