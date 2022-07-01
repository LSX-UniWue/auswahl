"""
======================
Benchmarking - Example
======================

Example demonstrating the feature selection benchmarking facilities.
TODO: extend

"""
import pickle

import numpy as np
import pandas as pd
from auswahl import MCUVE, CARS, VIP, IPLS, VIP_SPA, FiPLS
from benchmark import *

from sklearn.metrics import mean_squared_error, mean_absolute_error

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
                features=[(2, 10), (2, 11), (2, 12)],  # integer or tuple of integers (n_intervals, width) for IntervalSelectors
                n_runs=10,
                reg_metrics=[mean_squared_error],
                stab_metrics=[deng_score],
                methods=[mcuve, vip, cars, ipls, fipls],
                random_state=11111111,
                n_jobs=2,
                error_log_file="./error_log.txt",
                verbose=True)

# store benchmark data
pod.store(".", "pod")

#
# Plotting
#

# plot regression score (subset of methods and n_features specifiable)
plot_score(pod, save_path="./score.png")

# plot score/stability (subset of methods)
plot_score_vs_stability(pod, n_features=(2, 11), save_path="./score_stability.png")

# plot stability (subset of methods specifiable)
plot_stability(pod, save_path="./stability.png")

# plot execution time (subset of methods and n_features specifiable)
plot_exec_time(pod, item='median', save_path="./execution_time.png")

# plot selection probabilities for a specified number of features (subset of methods specifiable)
plot_selection(pod, n_features=(2, 11), save_path="./selection.png")


#
# extracting data
#

pod.get_regression_data(item="mean").to_csv("regression_means.csv")


#
# Storing and loading data
#

# store benchmark data
pod.store(".", "pod")

# load the benchmark data
pod = load_pod("./pod.pickle")











