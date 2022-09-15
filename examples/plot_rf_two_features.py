"""
===========================
Random Frog - Basic example
===========================

A Random Frog example showing the feature importance for a synthetic regression task.

The example uses a synthetic dataset with 10 standard normally distributed features.
The target values only depend on two features: #0 and #5.
If the Random Frog method is tasked with selecting two features, it identifies the two important features as shown below.

Note, that we use 1000 iterations to decrease the runtime in this case.
A small number of iterations likely results in unstable selections if no fixed random seed is given.
We recommend to use the default 10000 iterations when using the Random Frog method.
"""

import matplotlib.pyplot as plt
import numpy as np

from auswahl import RandomFrog

np.random.seed(1337)
X = np.random.randn(100, 10)
y = 5 * X[:, 0] - 2 * X[:, 5]

n_iterations = 1000
rf = RandomFrog(n_features_to_select=2, n_iterations=n_iterations, n_jobs=5, random_state=7331)
rf.fit(X, y)

colors = np.full(X.shape[1], fill_value='C00')
colors[rf.get_support()] = 'C01'

plt.bar(x=np.arange(X.shape[1]), height=rf.frequencies_ / n_iterations, color=colors)

plt.xlabel('Feature')
plt.ylabel('Relative Frequency')

plt.show()
