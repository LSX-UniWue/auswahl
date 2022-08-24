"""
=====================
FiPLS - Basic example
=====================

An Forward interval Partial Least Squares example showing the spectral intervals optimized by
the FiPLS method.

The example uses a synthetic dataset with 50 standard normally distributed features.
The target values only depend on four features: #21 and #24 and #46 and #47.
If the FiPLS method is tasked with selecting two intervals of width 5, it identifies
two intervals containing the above enumerated features.
"""

import matplotlib.pyplot as plt
import numpy as np

from auswahl import FiPLS

np.random.seed(1337)
X = np.random.randn(100, 50)
y = 5 * X[:, 21] - 2 * X[:, 24] + 3 * X[:, 46] + X[:, 47]

fipls = FiPLS(n_intervals_to_select=2, interval_width=5)
fipls.fit(X, y)

plt.bar(x=np.arange(X.shape[1]), height=fipls.get_support())

plt.xlabel('Feature')
plt.ylabel('Selection weight')

plt.show()
