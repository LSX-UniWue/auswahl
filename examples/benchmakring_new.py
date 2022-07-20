"""
======================
Benchmarking - Example
======================

Example demonstrating the feature selection benchmarking facilities.
TODO: extend

"""

import numpy as np
from sklearn.metrics import mean_squared_error

from auswahl import MCUVE, CARS, VIP, IPLS, VIP_SPA, FiPLS
from auswahl.benchmarking import benchmark, deng_score, plot_score, plot_score_vs_stability, plot_stability, \
    plot_exec_time, plot_selection, load_pod, zucknick_score

# load sample dataset
x = np.load("./data/spectra.npy")
y = np.load("./data/targets.npy")


#
# Selector definitions
# TODO-Remark: provide a valid default value for the n_features_to_select to avoid having to specify them (benchmark will override them anyway)
#

mcuve = MCUVE(n_features_to_select=10)
cars = CARS(n_features_to_select=10, n_jobs=2)

# demo model_hyperparams
vip = VIP(n_features_to_select=10, model_hyperparams={'n_components': [1, 2, 3]})

ipls = IPLS(interval_width=10, n_jobs=2)
vip_spa = VIP_SPA(n_features_to_select=10, n_jobs=2)
ipls = IPLS(interval_width=10, n_jobs=2)
fipls = FiPLS(interval_width=10)


pod = benchmark((x, y, 'nitrogen', 0.9),  # spectrum, target, dataset name, train size
                features=[(2, 2), (3, 8), (2, 12)],  # integer or tuple of integers (n_intervals, width) for IntervalSelectors
                n_runs=10,
                reg_metrics=mean_squared_error,
                stab_metrics=zucknick_score,
                methods=[mcuve, vip],
                random_state=11111111,
                n_jobs=2,
                error_log_file="./error_log.txt",
                verbose=True)


#print(pod.get_selection_data(n_features=(2, 2), sample=[0]))
plot_score_vs_stability(pod, n_features=(2, 12), save_path="./score_stability.png")
#print(pod.get_regression_data(n_features=[(2, 2), (2, 20)]))
#plot_exec_time(pod, save_path="./execution_time.png")
#plot_score(pod, save_path="./score.png")
#plot_stability(pod, save_path="./stability.png")
#plot_selection(pod, n_features=(2, 12), save_path="./selection.png")