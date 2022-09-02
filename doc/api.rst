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

Benchmarking
============

.. autosummary::
    :toctree: generated/
    :template: function.rst

    benchmarking.benchmark
    benchmarking.load_data_handler

.. autosummary::
    :toctree: generated/
    :template: class.rst

    auswahl.Selection
    benchmarking.DataHandler


