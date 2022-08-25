AUSWAHL documentation
=====================

A scikit-learn compatible package for wavelength selection.

Introduction
------------

AUSWAHL (AUtomatic Selection of WAvelengtH Library) is a scikit-learn compatible python package that provides a
collection of feature selection methods that are intended for spectral datasets.

The goal of AUSWAHL is to provide a unified implementation of several feature selection methods that are popular in the
field of chemometrics. Many methods exist (often designed for near-infrared spectra) but comparing them is
difficult without a shared code basis.

Using scikit-learn as the underlying framework, allows to integrate the methods from AUSWAHL in your existing
machine learning projects. Therewith, all methods known from scikit-learn (e.g. pipelines, cross-validation,
hyperparameter search) can also be used with the feature selection methods from the AUSWAHL package.
The feature selection methods can be executed by simply calling ``selector.fit(X, y)``.
Afterwards, the selections are retrieved by calling ``selector.get_support()``.

Another goal of AUSWAHL is to provide a large set of feature selection methods as a benchmark for researchers that
develop novel feature selection methods.

If you use AUSWAHL, please cite our work:

.. code:: bibtex

    @manual{auswahl,
        author = {Florian Buckermann and Stefan Heil and Anna Krause},
        title = {AUSWAHL - A scikit-learn compatible package for wavelength selection},
        month = mar,
        year = 2022
    }

.. toctree::
    :maxdepth: 3

    installation
    quickstart

.. toctree::
    :maxdepth: 3
    :caption: User Guide

    user_guide

.. toctree::
    :maxdepth: 3
    :caption: Documentation

    api

.. toctree::
    :caption: Examples

    auto_examples/index
