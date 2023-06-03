# BiSPy : Bivariate Signal Processing with Python

[![Documentation](https://readthedocs.org/projects/bispy/badge/?style=default)](https://bispy.readthedocs.org)
[![Build Status](https://travis-ci.org/jflamant/bispy.svg?branch=master)](https://travis-ci.org/jflamant/bispy)

BiSPy is an open-source python framework for processing bivariate signals. It supports our papers on time-frequency analysis [1], spectral analysis [2] and linear time-invariant filtering [3] of bivariate signals.

> [1] Julien Flamant, Nicolas Le Bihan, Pierre Chainais: “Time-frequency analysis of bivariate signals”, In press, Applied and Computational Harmonic Analysis, 2017; [arXiv:1609.0246](http://arxiv.org/abs/1609.02463), [doi:10.1016/j.acha.2017.05.007](https://doi.org/10.1016/j.acha.2017.05.007)

> [2] Julien Flamant, Nicolas Le Bihan, Pierre Chainais: “Spectral analysis of stationary random bivariate signals”, 2017, IEEE Transactions on Signal Processing; [arXiv:1703.06417](http://arxiv.org/abs/1703.06417), [doi:10.1109/TSP.2017.2736494](https://doi.org/10.1109/TSP.2017.2736494)

> [3] Julien Flamant, Pierre Chainais, Nicolas Le Bihan: “A complete framework for linear filtering of bivariate signals”, 2018; Accepted for publication in IEEE Transactions on Signal Processing; [arXiv:1802.02469](https://arxiv.org/abs/1802.02469)

These papers contains theoretical results and several applications that can be reproduced with this toolbox.


This python toolbox is currently under development and is hosted on GitHub. If you encounter a bug or something unexpected please let me know by [raising an issue](https://github.com/jflamant/bispy/issues) on the project page.

Requirements
============
BiSPy works with python 3.5+.

Dependencies:
 -   [NumPy](http://www.numpy.org)
 -   [SciPy](https://www.scipy.org)
 -   [Matplotlib](http://matplotlib.org)
 -   [numpy-quaternion](https://github.com/moble/quaternion)

To install dependencies:
```shell
pip install numpy scipy matplotlib numpy-quaternion
```

[quaternion](https://github.com/moble/quaternion) add quaternion dtype support to numpy. Implementation by [moble]. Since this python toolbox relies extensively on this module, you can check out first the nice introduction [here](https://github.com/moble).

### Install from sources

Clone this repository

```bash
git clone https://github.com/jflamant/bispy.git
cd bispy
```

And execute `setup.py`

```bash
pip install .
```

License
=======
This software is distributed under the [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

Cite this work
==============
If you use this package for your own work, please consider citing it with this piece of BibTeX:

```bibtex
@misc{BiSPy,
    title =   {{BiSPy: an Open-Source Python project for processing bivariate signals}},
    author =  {Julien Flamant},
    year =    {2018},
    url =     {https://github.com/jflamant/bispy/},
    howpublished = {Online at: \url{github.com/jflamant/bispy/}},
    note =    {Code at https://github.com/jflamant/bispy/, documentation at https://bispy.readthedocs.io/}
}
```
