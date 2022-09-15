.. currentmodule:: auswahl

.. |br| raw:: html

  <div style="line-height: 0; padding: 0; margin: 0"></div>

============
Benchmarking
============



Data sets
=========

The benchmarking system can evaluate the performance of :class:`SpectralSelector` methods across several data sets simultaneously in order to provide a
unified comparison of selection algorithms across a number of studied scenarios.
The data set configurations provided to the benchmarking function each consist of a tuple specifiying four items. Namely

    *  The spectral data as :class:`numpy.ndarray`
    * The target quantities as :class:`numpy.ndarray`
    * The unique name of the data set
    * a float in ]0,1[ indicating the share of the data to be used for training

An example invocation of :func:`benchmarking.benchmark` with several data sets is given below::

    result = benchmark(data=[(X, y, 'first dataset', 0.90),
			                 (X2, y2, 'second_dataset')],
                       features=[10, 15, 20],
                       methods=[CARS(), VIP()],
                       n_runs=10)


Selectors
=========

The benchmarking system can handle all selectors extending the class :class:`SpectralSelector`. Especially, the system can benchmark algorithms of
:class:`PointSelector` and :class:`IntervalSelector` simultaneously. In the data handling of the benchmarking system
and all derived functionalities, such as the plotting, the selectors are addressed by their class name. If these are not unique
(for instance during benchmarking of differently configured instances of the same selector algorithm),
or custom names are desired for other reasons, the names have to be specified for the selectors as exemplified below::

    result = benchmark([(X, y, 'data_example', 0.25)],
                       features=[10, 15, 20],
                       # provide unique names here
                       methods=[(CARS(), "first_cars"), (CARS(n_cars_runs=10), "second_cars")],
                       n_runs=10)


Features to select
==================

The benchmarking system allows the comparison of feature selection algorithms across several feature configurations.
For the :class:`PointSelector` the configurations of features to be selected are simply specified by a single integer. For selectors extending :class:`IntervalSelector`
a feature selection configuration requires the description of both the width of an interval (that is a consecutive block of wavelengths) and a number of
of such intervals to be extracted by the algorithm. Such a configuration is specified with a tuple (number of intervals, interval width). If the methods to be benchmarked
comprise at least one selector extending :class:`IntervalSelector`, all feature configurations need to be specified in the above defined interval fashion. For
:class:`PointSelector` the interval configurations will be resolved to a number of interval width times number of intervals individual features to be selected.
An example for is given below::

    result = benchmark([(X, y, 'data_example', 0.25)],
			            #The IntervalSelector FiPLS is benchmarked. Specify the feature configurations for which the algorithms
			            #are to be tested as intervals (#intervals, interval width)
                       	features=[(10, 5), (20, 5), (5, 10)],
                        methods=[CARS(), FiPLS()],
                        n_runs=10)


Metrics
=======

The benchmarking system distinguishes between two different kinds of metrics, namely *regression metrics* and *stability metrics*.
While the former is a compulsory component of the benchmarking (and set to :func:`sklearn.metrics.mean_squares_error` per default),
the *stability metric* is optinally calculated by the benchmarking system. The system can handle several metrics of each kind simultaneously.

Regression Metrics
------------------

The benchmarking system can handle all regression metrics, which follow the convention of the metric functions of sklearn, such as
:func:`sklearn.metrics.mean_squared_error` or :func:`sklearn.metrics.mean_absolute_error`.

Stability Metrics
-----------------

The *stability* is a metric quantifying the degree of feature fluctuation of feature selection
algorithms across several executions with equal or varying data splits.
The stability is a well-established characteristic considered
in the feature selection literature and often juxtaposed with the regression quality of the features selected by an algorithm.
It can be argued, that the stability of a selection algorithm is a confluence of both the amount of randomization deployed in the algorithm and
the degree to which randomization is modulated by the algorithm through feedback from interactions with the data,
which might allow for many or few different sets of features with high explanatory power, depending on its characteristics.
Some of these characteristics of data are innate to their acquisition domain.
An ubiquitous property of spectral data is its high degree of multicollinearity, that is a high degree of linear dependence und redundancy
between different features. The more conventional stability metrics, which are based on set-theoretical considerations such as the Intersection-over-Union can therefore be considered as not entirely adequate for the stability assessment
in the regime of spectral feature selection and many other domains. The set-theoretical approaches have therefore been
complemented with metrics introducing correlation-adjustment mechanisms into the stability
evaluation. |br|
For the benchmarking functionality provided in *Auswahl*, two stability assessment metrics are provided directly.

Deng Score
^^^^^^^^^^

The Deng-score is a set-theoretical stability measure without any means of correlation-adjustment.
The metric considers two sets of selections of features :math:`S_1` and :math:`S_2` of size :math:`n` out of :math:`N` features
and quantifies the degree of overlap adjusted by the expected random overlap :math:`e = \frac{n^2}{N}`

.. math:: \mathcal{S}_{deng}(S_1, S_2) = \frac{S_1 \cap S_2 - e}{n - e}


For sets of selections :math:`\{S_i\}_{i=1}^k`, the score is averaged across all pairs

.. math:: \mathcal{S}_{deng}(\{S_i\}_{i=1}^k) = \frac{2}{k^2 - k}\displaystyle\sum_{i < j}\mathcal{S}(S_i, S_j)


The metric is available for benchmarking with function :func:`benchmarking.util.metrics.deng_score`

.. topic:: References:

    * Bai-Chuan Deng, Yong-Huan Yun, Pan Ma, Chen-Chen Li, Da-Bing Ren and Yi-Zeng Liang,
      'A new method for wavelength interval selection that intelligently optimizes the locations, widths
      and combination of intervals',
      Analyst, 6, 1876-1885, 2015.

Zucknick Score
^^^^^^^^^^^^^^

The Zucknick score aims to account for the high collinearity in spectral data, by incorporating a corrletion adjustment mechanism into the stability evaluation.
To that end the Zucknick-Score consideres the Intersection-over-Union adjusted with a correlation contribution :math:`C`. The metric considers two sets of selections of features :math:`S_1` and :math:`S_2` of size :math:`n`:

.. math:: \mathcal{S(\delta)}_{zucknick} = \frac{S_1 \cap S_2 + C(S_1, S_2, \delta)}{S_1 \cup S_2}

, where :math:`C` is defined as

.. math:: C(S_1, S_2, \delta) = f(S_1, S_2, \delta) + f(S_2, S_1, \delta)

, where in turn :math:`f` is defined as

.. math:: f(S_1, S_2, \delta) =
            \begin{cases}
                0       & \quad \text{if } |S_1 \backslash S_2| = 0\\
                \frac{\|X(S_1 \backslash S_2, S_2) \odot t(X(S_1 \backslash S_2, S_2), \delta)\|}{n}  & \quad \text{else}
            \end{cases}

, where :math:`X(S_1, S_2)` is the correlation matrix between the features in :math:`S_1` and :math:`S_2`, :math:`\odot` the elementwise multiplication
and :math:`t(X, d)` a thresholding function operating on matrix :math:`X` using threshold :math:`d`, such that

.. math:: t(X, d)_{i,j} = \begin{cases}
    				0       & \quad \text{if } X_{ij} < d\\
    				X_{ij}  & \quad \text{else}
  			  \end{cases}

The parameter :math:`\delta` can be selected by the users as a threshold for the minimum required correlation between
two features to consider them similar.

.. topic:: References:

    * Zucknick, M., Richardson, S., Stronach, E.A.: Comparing the characteristics of
      gene expression profiles derived by univariate and multivariate classification methods.
      Stat. Appl. Genet. Molecular Biol. 7(1), 7 (2008)

Adding Stability Metrics
^^^^^^^^^^^^^^^^^^^^^^^^

In order to add custom stability metrics to the benchmarking system, consider the documentation for the
helper function :func:`benchmarking.util.metrics.pairwise_scoring`.

Plotting Facilities
===================



