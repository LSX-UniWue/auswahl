"""
==================================
IntervalRandomFrog - Basic example
==================================

An Interval Random Frog example showing the spectral intervals optimized by
the IRF method.

The example uses a synthetic dataset with 50 standard normally distributed features.
The target values only depend on four features: #21 and #24 and #46 and #47.
If the Interval Random Frog method is tasked with selecting two intervals of width 5, it identifies
two intervals containing the above enumerated features.

Note, that we use 1000 iterations to decrease the runtime in this case.
A small number of iterations likely results in unstable selections if no fixed random seed is given.
We recommend to use the default 10000 iterations when using the Random Frog method.
"""

import matplotlib.pyplot as plt
import numpy as np

from auswahl import IntervalRandomFrog

np.random.seed(1337)
X = np.random.randn(100, 50)
y = 5 * X[:, 21] - 2 * X[:, 24] + 3 * X[:, 46] + X[:, 47]

n_iterations = 1000
irf = IntervalRandomFrog(n_intervals_to_select=2,
                         interval_width=5,
                         n_iterations=n_iterations,
                         n_jobs=5,
                         random_state=7331)
irf.fit(X, y)

idx = np.arange(len(irf.frequencies_))
plt.plot(idx, irf.frequencies_ / n_iterations, marker='.', zorder=3)
plt.hlines(y=irf.frequencies_ / n_iterations,
           xmin=idx,
           xmax=idx + irf.interval_width - 1,
           alpha=0.5,
           zorder=1)

interval_starts = np.argwhere(np.diff(irf.get_support().astype(int)) > 0) + 1
plt.hlines(y=irf.frequencies_[interval_starts] / n_iterations,
           xmin=interval_starts,
           xmax=interval_starts + irf.interval_width - 1,
           colors='C01',
           zorder=2)

plt.ylim([0, 1.01])
plt.grid(axis='y')
plt.xticks(range(0, 55, 5))
plt.xlabel('Feature')
plt.ylabel('Relative Frequency')
plt.legend(['Frequency', 'Interval', 'Selected Intervals'])

plt.show()
