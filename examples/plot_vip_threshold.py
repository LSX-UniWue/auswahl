"""
================================
VIP - Selection with a threshold
================================

A Variable Importance in Projection example showing how to select features without defining the number of features to
select.

Usually the number of features to select is not known before.
The VIP method computes a score for each feature, and thus we can use these scores in combination with a threshold to
select the most important features.

The example below uses a synthetic dataset with 10 standard normally distributed features.
The target values only depend on three features: #0, #5 and #8.
We can use the :func:`get_support_for_threshold <auswahl.VIP.get_support_for_threshold>` method to select features
whose VIP values are higher than a given threshold.
A common threshold is 1 which leads to the correct selection of the three important features in this case.

.. note::
    See also :ref:`sphx_glr_auto_examples_plot_vip_two_features.py`

"""
import matplotlib.pyplot as plt
import numpy as np

from auswahl import VIP

np.random.seed(1337)
X = np.random.randn(100, 10)
y = X[:, 0] - X[:, 5] + X[:, 8]

vip = VIP()
vip.fit(X, y)

selection = vip.get_support_for_threshold(threshold=1)
colors = np.full(X.shape[1], fill_value='C00')
colors[selection] = 'C01'

plt.bar(x=np.arange(X.shape[1]), height=vip.vips_, color=colors)
plt.axhline(1, linestyle='--', color='black')
plt.xlabel('Feature')
plt.ylabel('VIP')

plt.show()
