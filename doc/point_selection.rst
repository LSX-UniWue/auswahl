.. currentmodule:: auswahl

.. _base.PointSelector:

==========================
Wavelength Point Selection
==========================

All classes that extend the :class:`auswahl.PointSelector` can be used to perform wavelength point selection,
i.e. the independent selection of informative wavelengths without taking spatial information into account.

Note that neither the number of features to select nor the hyperparameters of the different models are optimized
according to your use-case.
Use the methods available in :mod:`sklearn.model_selection` to determine a suitable set of parameters.

.. _vip:

Variable Importance in Projection
=================================

Calculating the Variable Importance in Projection [v]_ (VIP) is possible by using the :class:`VIP` selection method.
This method uses the coefficients of a fitted Partial Least Squares (PLS) model to calculate the importance of each
wavelength (=variable).
The VIP for wavelength j is computed by

.. math:: VIP_j = \sqrt{N \frac
          {\sum_k \left((b_k^2 \mathbf{t}_k^T \mathbf{t}_k)(\mathbf{w}_{jk}/||\mathbf{w}_k||)^2\right)}
          {\sum_k (b_k^2 \mathbf{t}_k^T \mathbf{t}_k)}}

.. topic:: Examples

    TODO

.. topic:: References:

   .. [v] Stefania Favilla, Caterina Durante, Mario Li Vigni, Marina Cocchi,
      'Assessing feature relevance in NPLS models by VIP',
      Chemometrics and Intelligent Laboratory Systems, 129, 76--86, 2013.

.. _mcuve:

Monte-Carlo Uninformative Variable Elimination
==============================================

Monte-Carlo Uninformative Variable Elimination (MC-UVE) is available in :class:`MCUVE`.

.. _rf:

Random Frog
===========

Random Frog (RF) is available in :class:`RandomFrog`.

.. _cars:

Competitive Adaptive Reweighted Sampling
========================================

Competitive Adaptive Reweighted Sampling (CARS) is available in :class:`CARS`.

.. _spa:

Successive Projection Algorithm
===============================

Successive Projection Algorithm (SPA) is available in :class:`SPA`.