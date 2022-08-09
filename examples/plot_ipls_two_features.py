"""
====================
IPLS - Basic example
====================

An Interval Partial Least Squares example showing the optimal interval of specified size for the regression of synthetic
data.

The example uses a synthetic dataset with 50 standard normally distributed features.
The target values only depend on two features: #21 and #24.
If the VIP method is tasked with selecting an interval of width 5, it identifies
an interval containing the features #21 and #24
"""
import matplotlib.pyplot as plt
import numpy as np

from auswahl import IPLS

np.random.seed(1337)
X = np.random.randn(100, 50)
y = 5 * X[:, 21] - 2 * X[:, 24]

ipls = IPLS(interval_width=5)
ipls.fit(X, y)

plt.bar(x=np.arange(X.shape[1]), height=ipls.get_support())

plt.xlabel('Feature')
plt.ylabel('Selection weight')

plt.show()
