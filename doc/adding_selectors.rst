
.. currentmodule:: auswahl

.. |br| raw:: html

  <div style="line-height: 0; padding: 0; margin: 0"></div>

===========================
Adding Selectors to Auswahl
===========================

In general the extension of Auswahl with custom selector algorithms only requires extending the appropriate base class :class:`~auswahl.PointSelector`
or :class:`~auswahl.IntervalSelector` and their proper initialization. The implementation of the member function :meth:`~auswahl.PointSelector._fit` is required.
After the selector has been fitted on data, the instance must however possess the attribute ``support_``, an array of flags indicating the selected features, and
the attribute ``best_model_``, which is an instance of the regression model underlying the selector fitted on the selected features. |br|

The baseclass :class:`~auswahl.SpectralSelector` provides the method :meth:`~auswahl.SpectralSelector.evaluate`, which
allows easy evaluation of the underlying regressor models on data with projections on different features. A cross-validation
and hyperparameter optimization of the regressor can be optionally conducted.

PointSelector to IntervalSelector conversion
============================================

If the custom selector algorithm does not only calculate a specific selection of features, but
produces a weight (greater better) for each feature, the selector can be made accessible to
an algorithmic converion from a :class:`~auswahl.PointSelector` to :class:`~auswahl.IntervalSelector` by making the custom selector additionally extend
the class :class:`~auswahl.Convertible`. The extension requires the implementation of the member function :meth:`~auswahl.Convertible.get_feature_scores`.
The custom selector can then be passed as argument to an instance of :class:`~auswahl.PseudoIntervalSelector`. An example is given for the :class:`~auswahl.Convertible`
extending selector :class:`~auswahl.VIP`::

	from auswahl import VIP, PseudoIntervalSelector

	vip = VIP()
	interval_vip = PseudoIntervalSelector(selector=vip, n_intervals_to_select=10, interval_width=5)

The conversion is handled by a dynamic program optimizing the placements of intervals into the range of features, which maximizes the overall weight of
the features covered by the intervals.

Preparing the selector for benchmarking
=======================================

If the custom selector is to be used in the function :func:`~auswahl.benchmarking.benchmark` further implementations and overridings
are to be considered. If the selector comprises a composite internal structure, that is, if it contains other selector algorithms as subalgorithms
in its own selection approach, the following functions need to be overridden

* :meth:`~auswahl.SpectralSelector.rethread`
* :meth:`~auswahl.SpectralSelector.reseed`
* :meth:`~auswahl.SpectralSelector.reparameterize`

An example of such overrides can be seen in :class:`~auswahl.VIP_SPA`.