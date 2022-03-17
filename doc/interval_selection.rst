.. currentmodule:: auswahl

.. _base.IntervalSelector:

All classes that extend the :class:`auswahl.IntervalSelector` can be used to perform wavelength interval selection,
i.e. the selection of ranges of informative wavelengths.

In general, the approaches provide the number of interval selected and the width of the intervals as parameters.
The user is warned, if individual algorithms conceptually predetermine one of these parameters.
Note, that neither of these parameters, nor the hyperparameters of the deployed regression models are optimized directly.
Use the methods available in :mod:`sklearn.model_selection` to determine a suitable set of parameters.

.. _ipls:

Inteval Partial Least Squares
=================================

Internal Partial Least Squares is available in :class:`IPLS`.
IPLS is a simple algorithm, selecting the best interval of a user definable width w.r.t. to
a regression model.