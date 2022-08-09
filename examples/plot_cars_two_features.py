"""
====================
CARS - Basic example
====================

A Competetive Adaptive Reweighted Sampling example showing the feature importance defined as selection probability of
multiple CARS runs for a synthetic regression task.

The example uses a synthetic dataset with 10 standard normally distributed features.
The target values only depend on two features: #0 and #5.
"""
import matplotlib.pyplot as plt
import numpy as np

from auswahl import CARS

np.random.seed(1337)
X = np.random.randn(10, 10)
y = 5 * X[:, 0] - 2 * X[:, 5]

cars = CARS(n_features_to_select=2)
cars.fit(X, y)

colors = np.full(X.shape[1], fill_value='C00')
colors[cars.get_support()] = 'C01'

plt.bar(x=np.arange(X.shape[1]), height=cars.feature_importance_, color=colors)

plt.xlabel('Feature')
plt.ylabel('CARS')

plt.show()
