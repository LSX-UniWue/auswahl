.. currentmodule:: auswahl

===
API
===

Baseclasses
===========

.. autosummary::
    :toctree: generated/
    :template: class.rst

    FeatureDescriptor
    SpectralSelector
    PointSelector
    IntervalSelector
    Convertible

Wavelength Point Selection
==========================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    PointSelector
    CARS
    MCUVE
    RandomFrog
    SPA
    VIP

Wavelength Interval Selection
=============================
.. autosummary::
    :toctree: generated/
    :template: class.rst

    IntervalSelector
    IPLS
    FiPLS
    BiPLS
    IntervalRandomFrog
    PseudoIntervalSelector

Utilities
=========

.. autosummary::
    :toctree: generated/
    :template: function.rst

    optimize_intervals
    util.get_coef_from_pls

================
Benchmarking API
================

Benchmarking
============

.. autosummary::
    :toctree: generated/
    :template: function.rst

    benchmarking.benchmark

Data Handling
=============

.. autosummary::
    :toctree: generated/
    :template: function.rst

    benchmarking.load_data_handler

    :template: class.rst

    benchmarking.DataHandler
    benchmarking.util.helpers.Selection

Plotting
========

.. autosummary::
    :toctree: generated/
    :template: function.rst

    benchmarking.plot_score
    benchmarking.plot_stability
    benchmarking.plot_score_vs_stability
    benchmarking.plot_selection
    benchmarking.plot_exec_time

Stability Metrics
=================

.. autosummary::
    :toctree: generated/
    :template: class.rst

    benchmarking.util.metrics.StabilityScore
    benchmarking.util.metrics.PairwiseStabilityScore
    benchmarking.util.metrics.DengScore
    benchmarking.util.metrics.ZucknickScore

Misc
====

.. autosummary::
    :toctree: generated/
    :template: function.rst

    benchmarking.util.helpers.load_data_handler




