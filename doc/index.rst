.. BiSPy documentation master file, created by
   sphinx-quickstart on Tue Jan 31 16:28:23 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BiSPy's documentation!
===================================
BiSPy is an open-source python framework for signal processing of bivariate signals. It comes in addition of the paper

-  Julien Flamant, Nicolas Le Bihan, Pierre Chainais: “Time-frequency
   analysis of bivariate signals”, 2016;
   `arXiv:1609.0246 <http://arxiv.org/abs/1609.02463>`_ 

The paper contains theoretical results and several applications that can
be reproduced with this toolbox.

This python toolbox is currently under development and is hosted on GitHub. If you encounter a bug or something unexpected please let me know by raising an issue on the project page. 

Requirements
------------
 - `NumPy <http://www.numpy.org>`_
 - `SciPy <https://www.scipy.org>`_
 - `Matplotlib <http://matplotlib.org>`_
 - `quaternion <https://github.com/moble/quaternion>`_ 

 `quaternion <https://github.com/moble/quaternion>`_  add quaternion dtype support to numpy. Implementation by `moble <https://github.com/moble>`_. Since this python toolbox relies extensively on this module, you can check out first the nice introduction `here <https://github.com/moble/quaternion>`_.


Documentation contents
----------------------

.. toctree::
   :maxdepth: 2

   tutorials/index
   reference/index

Licence
-------
This software is distributed under the `CeCILL Free Software Licence Agreement <http://www.cecill.info/licences/Licence_CeCILL_V2-en.html>`_. 

