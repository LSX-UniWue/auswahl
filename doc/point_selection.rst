.. currentmodule:: auswahl

.. _base.PointSelector:

All classes that extend the :class:`auswahl.PointSelector` can be used to perform wavelength point selection,
i.e. the independent selection of informative wavelengths without taking spatial information into account.

Note that neither the number of features to select nor the hyperparameters of the different models are optimized
according to your use-case.
Use the methods available in :mod:`sklearn.model_selection` to determine a suitable set of parameters.

.. _vip:

Variable Importance in Projection
=================================

Calculating the Variable Importance in Projection (VIP) is possible by using the :class:`VIP` selection method.
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

    * :ref:`sphx_glr_auto_examples_plot_vip_two_features.py`: A VIP example usage for a synthetic regression task
    * :ref:`sphx_glr_auto_examples_plot_vip_threshold.py`: A VIP example that examines the calculated VIP scores to
      determine the number of features to select

.. topic:: References:

   * Stefania Favilla, Caterina Durante, Mario Li Vigni, Marina Cocchi,
     'Assessing feature relevance in NPLS models by VIP',
     Chemometrics and Intelligent Laboratory Systems, 129, 76--86, 2013.

.. _mcuve:

Monte-Carlo Uninformative Variable Elimination
==============================================

Monte-Carlo Uninformative Variable Elimination (MC-UVE) uses random sampling to determine the stability of features.
Performing MC-UVE is possible by using the :class:`MCUVE` selection method.
The method generates a large number of random subsets of the training data and fits a PLS model to each subset.
Afterwards, the importance of each feature is determined by computing the stability of the PLS' regression coefficients.
If μ and σ are the mean and standard deviation of the regression coefficients,
the stability for the i-th feature is defined as:

.. math:: s_i = \frac{\mu_i}{\sigma_i}

The features with the highest **absolute** stability values are selected.

.. topic:: References:

    * Wensheng Cai, Yankun Li and Xueguang Shao,
      'A variable selection method based on uninformative variable elimination for multivariate calibration of
      near-infrared spectra',
      Chemometrics and Intelligent Laboratory Systems, 90, 188-194, 2008.

.. _rf:

Random Frog
===========

Random Frog (RF) is available in :class:`RandomFrog`.

.. topic:: References:

    * Hong-Dong Li and Qing-Song Xu and Yi-Zeng Liang,
      'Random frog: An efficient reversible jump Markov Chain Monte Carlo-like approach for variable selection with
      applications to gene selection and disease classification',
      Analytica Chimica Acta, 740, 20-26, 2012

.. _cars:

Competitive Adaptive Reweighted Sampling
========================================

Competitive Adaptive Reweighted Sampling (CARS) is available in :class:`CARS`.
CARS is an iterative algorithm, producing a candidate feature set for each iteration. The best feature set is
determined among the candiate sets of required size using Cross Validation.

The CARS algorithm combines a competitive sampling of features with a scheduled shrinkage of the number of features
selected:

CARS considers an EDF (Exponential Decreasing function) to produce the upper bounds of features selected for the candiate set in each iteration, starting
with all features in the first and terminating with only 2 features in the final iteration.
The importance of features is quantified using their absolute regression weights in a fitted Partial Least Squares (PLS) model. The importance is subsequently
used as fitness in a competitive sampling procedure, in which the set of features is generated through sampling with replacement. The number of samples
drawn is determined by the EDF.

.. topic:: References:

    * Hongdong Li,Yizeng Liang, Qingsong Xu and Dongsheng Cao,
      'Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration',
      Analytica Chimica Acta, 648, 77-84, 2009

.. _spa:

Successive Projection Algorithm
===============================

Successive Projection Algorithm (SPA) is available in :class:`SPA`.
The SPA algorithm addresses the frequent problem of high collinearity in spectroscopic data:

If N is the number of samples and M the number of features, the algorithm considers the features as M
points in an N-dimensional vector space and conducts an iterative selection, by choosing features with
minimal lengths of projections onto the so far selected features. As a result, SPA constructs sets with minimum
collinearity.
Note, that SPA can only meaningfully construct sets of at most N features, which is the upper bound
of linearly independent features.

The iterative scheme of the algorithm makes the initial variable selected a degree of freedom.
Therefore, SPA considers every variable as candidate seed and subsequently selects the variable set with
a maximum CV performance.

Note also, that the features are selected solely with regard to their collinearity. The quality for
the target quantity regression is only considered during the CV optimization of the initial variable.

.. topic:: References:

    * Mário César Ugulino Araújo,Teresa Cristina Bezerra Saldanha, Roberto Kawakami Harrop Galvao,
      Takashi Yoneyama, Henrique Caldas Chame and Valeria Visani,
      'The successive projections algorithm for variable selection in spectroscopic multicomponent analysis',
      Chemometrics and Intelligent Laboratory Systems, 57, 65-73, 2001