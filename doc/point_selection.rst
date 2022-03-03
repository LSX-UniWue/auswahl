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
wavelength (=variable). To calculate the VIPs, the X-score matrix T, the y-loading vector q and the normalized X-weight
matrix W are used. They are defined as:

.. math:: \mathbf{T} \in \mathbb{R}^{N \times K},
          \mathbf{q} \in \mathbb{R}^{1 \times K},
          \mathbf{W} \in \mathbb{R}^{M \times K}

where N is the number of samples, M is the number of features and K is the number of latent variables.
The VIPs are computed by:

.. math:: \mathit{VIP} = \sqrt{M \frac{\mathbf{W}^2\left[\mathbf{q}^2\mathbf{T}^t\mathbf{T}\right]^t}
                       {\sum_k \left[\mathbf{q}^2\mathbf{T}^t\mathbf{T}\right]_k}}

Note that the above equation uses a 1-dimensional target vector.

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