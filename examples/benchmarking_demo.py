
"""
======================
Benchmarking - Example
======================

Example demonstrating the feature selection benchmarking facilities.
"""

import numpy as np
from sklearn.metrics import mean_squared_error

from auswahl import MCUVE, CARS, VIP, IPLS, VIP_SPA, FiPLS
from auswahl.benchmarking import benchmark, deng_score, plot_selection, plot_score_vs_stability, zucknick_score

#
# load sample dataset
#
x = np.load("./data/spectra.npy")
y = np.load("./data/targets.npy")


#
# Selector definitions
#

mcuve = MCUVE()
cars = CARS()
ipls = IPLS()

# Define a model with hyperparameters
vip = VIP(n_features_to_select=10, model_hyperparams={'n_components': [1, 2, 3]})


pod = benchmark((x, y, 'nitrogen', 0.9),
                features=[(2, 2), (3, 8), (4, 5)],
                n_runs=10,
                reg_metrics=mean_squared_error,
                stab_metrics=[zucknick_score, deng_score],
                methods=[vip, mcuve, cars],
                random_state=11111111,
                n_jobs=2,
                error_log_file="./error_log.txt",
                verbose=True)


plot_score_vs_stability(pod, n_features=(3, 8), stability_metric='zucknick_score', save_path="./score_stability.png")

plot_selection(pod, n_features=(3, 8), save_path="./selection.png")
