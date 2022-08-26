"""
=====================
BiPLS - Basic example
=====================

An Backward interval Partial Least Squares example showing the spectral intervals optimized by
the BiPLS method.

The example uses a synthetic dataset with 50 standard normally distributed features.
The target values only depend on four features: #21 and #24 and #46 and #47.
If the BiPLS method is tasked with selecting two intervals of width 5, it identifies two intervals containing the above
enumerated features.

The BiPLS method stores the order of removed intervals in the ``rank_`` attribute, i.e. the interval with the lowest
relative rank has been removed first, the interval with the second-lowest relative rank has been removed second, and so
on.
The finally selected intervals have a relative rank of 1.
"""

import matplotlib.pyplot as plt
import numpy as np

from auswahl import BiPLS

np.random.seed(1337)
X = np.random.randn(100, 50)
y = 5 * X[:, 21] - 2 * X[:, 24] + 3 * X[:, 46] + X[:, 47]

interval_width = 5
bipls = BiPLS(n_intervals_to_select=2, interval_width=5)
bipls.fit(X, y)

plt.step(range(len(bipls.rank_)), bipls.rank_, where='post', zorder=3)
plt.axhline(1, color='C01', zorder=2)

plt.grid()
plt.xlabel('Feature')
plt.ylabel('Relative Rank')

plt.show()
