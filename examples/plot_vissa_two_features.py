"""
===================
VISSA - Basic example
===================

A Variable Iterative Subspace Shrinkage Approach example showing the feature importance for a synthetic regression task.

The example uses a synthetic dataset with 10 standard normally distributed features.
The target values only depend on two features: #0 and #5.

"""
import matplotlib.pyplot as plt
import numpy as np

from auswahl import VISSA

np.random.seed(1337)
X = np.random.randn(100, 10)
y = 5 * X[:, 0] - 2 * X[:, 5]

vissa = VISSA(n_features_to_select=2, n_submodels=200, n_jobs=2)
vissa.fit(X, y)

colors = np.full(X.shape[1], fill_value='C00')
colors[vissa.get_support()] = 'C01'

plt.bar(x=np.arange(X.shape[1]), height=vissa.weights_, color=colors)

plt.xlabel('Feature')
plt.ylabel('VIP')

plt.show()
