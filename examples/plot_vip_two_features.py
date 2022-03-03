"""
===================
VIP - Basic example
===================

A Variable Importance in Projection example showing the feature importance for a synthetic regression task.

The example uses a synthetic dataset with 10 standard normally distributed features.
The target values only depend on two features: #0 and #5.
If the VIP method is tasked with selecting two features, it identifies the two important features as shown below.

.. note::
    See also :ref:`sphx_glr_auto_examples_plot_vip_threshold.py`

"""
import matplotlib.pyplot as plt
import numpy as np

from auswahl import VIP

np.random.seed(1337)
X = np.random.randn(100, 10)
y = 5 * X[:, 0] - 2 * X[:, 5]

vip = VIP(n_features_to_select=2)
vip.fit(X, y)

colors = np.full(X.shape[1], fill_value='C00')
colors[vip.get_support()] = 'C01'

plt.bar(x=np.arange(X.shape[1]), height=vip.vips_, color=colors)

plt.xlabel('Feature')
plt.ylabel('VIP')

plt.show()
