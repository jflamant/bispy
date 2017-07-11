# BiSPy : a python framework for signal processing of bivariate signals

[![Documentation](https://readthedocs.org/projects/bispy/badge/?style=default)](https://bispy.readthedocs.org)

BiSPy is an open-source python framework for signal processing of bivariate signals. It comes in addition of the following papers

>   Julien Flamant, Nicolas Le Bihan, Pierre Chainais: “Time-frequency analysis of bivariate signals”, 2016; [arXiv:1609.0246](http://arxiv.org/abs/1609.02463)

> Julien Flamant, Nicolas Le Bihan, Pierre Chainais: “Spectral analysis of stationary random bivariate signals”, 2016; [arXiv:1703.06417](http://arxiv.org/abs/1703.06417)

These papers contains theoretical results and several applications that can be reproduced with this toolbox.

This python toolbox is currently under development and is hosted on GitHub. If you encounter a bug or something unexpected please let me know by [raising an issue](https://github.com/jflamant/bispy/issues) on the project page.

Requirements
============
BiSPy works with python 3.5+.

Dependencies:
 -   [NumPy](http://www.numpy.org)
 -   [SciPy](https://www.scipy.org)
 -   [Matplotlib](http://matplotlib.org)
 -   [quaternion](https://github.com/moble/quaternion)

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


Licence
=======
This software is distributed under the [CeCILL Free Software Licence Agreement](http://www.cecill.info/licences/Licence_CeCILL_V2-en.html) 
