"""
===================
SPA - Basic example
===================

A Successive Projections Algorithm example demonstrating the feature selection process
for a synthetic regression task.
Since SPA does not directly evaluate feature importance based on a regression metric, it is used in conjunction with
VIP to construct a set of a linearly independent set of features with a VIP score exceeding a threshold.

The example uses a synthetic dataset with 10 standard normally distributed features.
The target values only depend on two features: #0 and #5.
"""
import matplotlib.pyplot as plt
import numpy as np

from auswahl import SPA, VIP

np.random.seed(1337)
X = np.random.randn(100, 10)
y = 5 * X[:, 0] - 2 * X[:, 5]

vip = VIP()
spa = SPA(n_features_to_select=2)

vip.fit(X, y)

# Mask all features with a VIP score below a threshold
mask = vip.vips_ > 0.3
# Call SPA on masked data
spa.fit(X, y, mask=mask)

colors = np.full(X.shape[1], fill_value='C00')
colors[spa.get_support()] = 'C01'

plt.bar(x=np.arange(X.shape[1]), height=vip.vips_, color=colors)

plt.xlabel('Feature')
plt.ylabel('VIP')

plt.show()
