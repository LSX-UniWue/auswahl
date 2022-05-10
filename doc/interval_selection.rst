.. currentmodule:: auswahl

.. _base.IntervalSelector:

All classes that extend the :class:`auswahl.IntervalSelector` can be used to perform wavelength interval selection,
i.e. the selection of ranges of informative wavelengths.

In general, the approaches provide the number of interval selected and the width of the intervals as parameters.
The user is warned, if individual algorithms conceptually predetermine one of these parameters.
Note, that neither of these parameters, nor the hyperparameters of the deployed regression models are optimized directly.
Use the methods available in :mod:`sklearn.model_selection` to determine a suitable set of parameters.

.. _ipls:

Interval Partial Least Squares
==============================

Interval Partial Least Squares (IPLS) is available in :class:`IPLS`.
IPLS is a simple algorithm, selecting the best interval of a user definable width w.r.t. to
a regression model.

.. _irf:

Interval Random Frog
====================

Interval Random Frog (iRF) is available in :class:`IntervalRandomFrog`.

.. topic:: References:

    * Yong-Huan Yun and Hong-Dong Li and Leslie R. E. Wood and Wei Fan and Jia-Jun Wang and Dong-Sheng Cao and
      Qing-Song Xu and Yi-Zeng Liang,
      'An efficient method of wavelength interval selection based on random frog for multivariate spectral calibration',
      Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy, 111, 31-36, 2013