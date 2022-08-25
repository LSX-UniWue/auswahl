.. currentmodule:: auswahl

.. _base.IntervalSelector:

Wavelength Interval Selection
=============================

All classes that extend the :class:`auswahl.IntervalSelector` can be used to perform wavelength interval selection,
i.e. the selection of ranges of informative wavelengths.

In general, the approaches provide the number of interval selected and the width of the intervals as parameters.
The user is warned, if individual algorithms conceptually predetermine one of these parameters.
Note, that neither of these parameters, nor the hyperparameters of the deployed regression models are optimized directly.
Use the methods available in :mod:`sklearn.model_selection` to determine a suitable set of parameters.

.. _ipls:

Interval Partial Least Squares
------------------------------

Interval Partial Least Squares (IPLS) is available in :class:`IPLS`.
IPLS is a simple algorithm, selecting the best interval of a user definable width w.r.t. to
a regression model.

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_plot_ipls_two_features.py`: An IPLS example usage for a synthetic regression task

.. topic:: References:

    * L. Nogaard, A. Saudland, J. Wagner, J. P. Nielsen, L. Munck, S. B. Engelsen,
      'Interval Partial Least-Squares Regression (iPLS):
      A comparative chemometric study with an example from Near-Infrared Spectrocopy'
      Applied Spectrosopy, Volume 54, Nr. 3, 413--419, 2000.

.. _fipls:

Forward Interval Partial Least Squares
--------------------------------------

Forward interval Partial Least Squares (FiPLS) is available in :class:`FiPLS`.
FiPLS is a variant of IPLS that sequentially selects intervals based on the cross-validated RMSE of a fitted PLS model.
The idea is similar to :class:`sklearn.feature_selection.SequentialFeatureSelector` with :code:`direction='forward'` but
this method selects continuous sequences of features instead of single features.

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_plot_fipls_two_features.py`: An FiPLS example usage for a synthetic regression task

.. topic:: References:

    * Zou Xiaobo, Zhao Jiewen, Li Yanxiao,
      'Selection of the efficient wavelength regions in FT-NIR spectroscopy for determination of SSC of ‘Fuji’
      apple based on BiPLS and FiPLS models',
      Vibrational Spectroscopy, vol. 44, no. 2, 220--227, 2007.

.. _bipls:

Backward Interval Partial Least Squares
---------------------------------------

Backward interval Partial Least Squares (BiPLS) is available in :class:`BiPLS`.
BiPLS is a variant of IPLS that sequentially removes intervals based on the cross-validated RMSE of a fitted PLS model.
The idea is similar to :class:`sklearn.feature_selection.SequentialFeatureSelector` with :code:`direction='backward'`
but this method selects continuous sequences of features instead of single features.

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_plot_bipls_two_features.py`: A BiPLS example usage for a synthetic regression task

.. topic:: References:

    * Zou Xiaobo, Zhao Jiewen, Li Yanxiao,
      'Selection of the efficient wavelength regions in FT-NIR spectroscopy for determination of SSC of ‘Fuji’
      apple based on BiPLS and FiPLS models',
      Vibrational Spectroscopy, vol. 44, no. 2, 220--227, 2007.

.. _irf:

Interval Random Frog
--------------------

Interval Random Frog (iRF) is and adaption of the :ref:`rf` method that selects intervals instead of single features.
It is an iterative selection method that starts with randomly selected intervals which are adapted during the iteration
process.
Each iteration, a random sub- or superset is created and compared against the previously selected intervals by
cross-validation.
The iRF method keeps track of a counter for each interval and the counters for all intervals in the "winning" set
(i.e. higher cv score) are increased after each iteration.

Using arbitrary interval positions might result in overlapping intervals (i.e. offset is smaller than the interval
width) with similar or even equal selection probabilities. To prevent overlapping intervals for the final selection
mask, the intervals are determined sequentially; i.e. an interval is selected if it has the highest selection
probability AND does not overlap with the previously selected intervals.

iRF is available in :class:`IntervalRandomFrog`.

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_plot_irf_two_features.py`: An iRF example usage for a synthetic regression task

.. topic:: References:

    * Yong-Huan Yun and Hong-Dong Li and Leslie R. E. Wood and Wei Fan and Jia-Jun Wang and Dong-Sheng Cao and
      Qing-Song Xu and Yi-Zeng Liang,
      'An efficient method of wavelength interval selection based on random frog for multivariate spectral calibration',
      Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy, 111, 31-36, 2013