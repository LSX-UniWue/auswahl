.. -*- mode: rst -*-

AUSWAHL
============================================================

AUSWAHL (**AU**\tomatic **S**\election of **WA**\velengt\ **H** **L**\ibrary) is a python module
that provides a collection of Feature Selection Methods for Near-Infrared Spectroscopy.

Scope
-----
This library provides a collection of supervised feature selection methods for near-infrared spectra.
(Note, that features are also called variables of simply wavelengths in the field of chemometrics.)
It does **not** provide any preprocessing methods or calibration models.
All features selection methods have been implemented in compliance with the sklearn library.
Thus, the methods can be used in a sklearn pipeline before an estimator.

Methods
-------
The libraray provides methods for selecting wavelengths without spatial constraints and methods for selecting continuous bands of wavelengths.
The following methods are implemented:

- Wavelength Point Selection: VIP, MC-UVE, Random Frog, CARS, SPA
- Wavelength Interval Selection: iPLS, FiPLS, BiPLS, iRF

Installation
------------

AUSWAHL is currently not available through PyPi. We are working on releasing it in the near future. Right now, you can install it from source::

  git clone https://github.com/mcfloskel/auswahl.git
  cd auswahl
  pip install .


Usage
-----

The provided feature selection methods implement the ``SelectorMixin`` base class from sklearn and can by used in the same way as the sklearn feature selection methods. Below you can find a simple example for the ``VIP`` selection method::

  import numpy as np
  from auswahl import VIP
  
  x = np.random.randn(100, 10) # 100 samples and 10 features
  y = 5 * x[:,0] - 2 * x[:,5]  # y depends only on two features
  
  vip = VIP(n_features_to_select=2)
  vip.fit(x,y)
  
  vip.get_support()
  >>> array([True, False, False, False, False,  True, False, False, False, False])
