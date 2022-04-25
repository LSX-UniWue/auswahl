
"""

    Example demonstrating the feature selection benchmarking facilities.

    TODO: extend

"""
import numpy as np
from auswahl import MCUVE, CARS, VIP, IPLS, VIP_SPA
from benchmark import benchmark, stability_score, plot_score_stability_box

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error


x = np.load("./data/spectra.npy")
y = np.load("./data/targets.npy")

mcuve = MCUVE(n_features_to_select=10)
cars = CARS(n_features_to_select=10)
vip = VIP(n_features_to_select=10)
ipls = IPLS(interval_width=10)
vip_spa = VIP_SPA(n_features_to_select=10, n_jobs=2)

pod = benchmark(x,
                y,
                n_runs=5,
                train_size=0.9,
                test_model=PLSRegression(n_components=1),
                reg_metrics=[mean_squared_error],
                stability_metrics=[stability_score],
                methods=[mcuve, cars, vip, vip_spa],
                random_state=1111111)

plot_score_stability_box(pod,
                         'stability_score',
                         'mean_squared_error',
                         save_path="./benchmark_plot.png")
