===========
Quick Start
===========

The provided feature selection methods implement the :class:`SelectorMixin <sklearn.feature_selection.SelectorMixin>`
base class from the :mod:`sklearn.feature_selection` module  and can by used in the same way as other feature selection
methods from sklearn.

Below you can find a simple example for the :class:`VIP <auswahl.VIP>` selection method::

  import numpy as np
  from auswahl import VIP

  rs = np.random.RandomState(1337) # Sample seed for reproducibility
  x = rs.randn(100, 10)            # 100 samples and 10 features
  y = 5 * x[:,0] - 2 * x[:,5]      # y depends only on two features

  selector = VIP(n_features_to_select=2)
  selector.fit(x,y)

  selector.get_support()
  >>> array([True, False, False, False, False, True, False, False, False, False])

If you want to use another feature selection method, you can simply replace ``VIP(...)`` with any other method.

The :class:`VIP <auswahl.VIP>` method also allows to select the number of features after the model has been fitted.
An example is given below::

  vip = VIP(n_features_to_select=x.shape[1]-1)
  vip.fit(x,y)
  vip.get_support_for_threshold(threshold=1)
  >>> array([True, False, False, False, False, True, False, False, False, False])

