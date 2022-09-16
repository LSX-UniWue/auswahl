"""
======================
Benchmarking - Example
======================

Example demonstrating the feature selection benchmarking facilities.
"""

import numpy as np

from auswahl import VIP, MCUVE, CARS
from auswahl.benchmarking import benchmark, ZucknickScore, DengScore, plot_score, plot_score_vs_stability

np.random.seed(1337)
X = np.random.randn(100, 100)
y = 5 * X[:, 0] - 2 * X[:, 5]

vip = VIP()
mcuve = MCUVE()
cars = CARS()

result = benchmark([(X, y, 'data_example', 0.8)],
                   features=[i for i in range(1, 10)],
                   methods=[vip, mcuve, cars],
                   n_runs=10,
                   random_state=42,
                   stab_metrics=[DengScore(), ZucknickScore(correlation_threshold=0.8)],
                   n_jobs=5,
                   verbose=False)

plot_score(result)
plot_score_vs_stability(result, stability_metric='deng_score', n_features=5)
