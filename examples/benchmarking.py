
"""

    Example demonstrating the feature selection benchmarking facilities.

    TODO: extend

"""
import numpy as np
from auswahl import MCUVE, CARS, VIP, IPLS, VIP_SPA, VISSA, RandomFrog, iVISSA
from benchmark import *

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


x = np.load("./data/spectra.npy")
y = np.load("./data/targets.npy")
n = np.load("./data/nitrogen.npy")

mcuve = MCUVE(n_features_to_select=10)
cars = CARS(n_features_to_select=10, n_jobs=2)
vip = VIP(n_features_to_select=10)
ipls = IPLS(interval_width=10, n_jobs=2)
vip_spa = VIP_SPA(n_features_to_select=10, n_jobs=2)
rf = RandomFrog(n_features_to_select=10)
ipls = IPLS(n_intervals_to_select=1, interval_width=10, n_jobs=2)

ivissa = iVISSA(n_intervals_to_select=2, interval_width=10)

pod = benchmark([(x, n, 'nitrogen'),
                 ],#(x, y, 'corn')],
                n_features=[12, 11, 10],  # np.arange(10, 300, 5).tolist(),
                #n_intervals=[1, 1, 1],
                #interval_widths=[10, 11, 12],
                n_runs=5,
                train_size=0.9,
                test_model=PLSRegression(n_components=1),
                reg_metrics=[mean_squared_error],
                stab_metrics=[deng_score],
                methods=[cars, (cars, "cars_the_second")],
                random_state=11111111,
                verbose=True)



#print(pod.get_selection_data(method='VIP', n_features=10))
#print(pod.get_regression_data(method='VIP', dataset='manure', n_features=10))
#print(pod.get_measurement_data())

#plot_score_stability_box(pod, 'manure', 10, 'stability_score', 'mean_squared_error')

#plot_exec_time(pod, dataset='manure')
#print(pod.get_regression_data(dataset='manure', method='VIP', reg_metric='mean_squared_error', item='samples'))
#plot_performance_series(pod, dataset='manure', regression_metric='mean_squared_error', item='median', save_path="./performance.png")

#plot_score_stability_box(pod,
 #                        dataset='nitrogen',
  #                       n_features=10,
   #                      stability_metric='zucknick_score',
    #                     regression_metric='mean_squared_error')

#plot_score_stability_box(pod,
 #                        dataset='nitrogen',
  #                       n_features=10,
   #                      stability_metric='deng_score',
    #                     regression_metric='mean_squared_error')

#strata, p = mw_ranking(pod, regression_metric='mean_squared_error')

#plot_stability_series(pod, dataset='nitrogen', stability_metric='stability_score', save_path="./stability.png")

#plot_selection(pod, dataset='nitrogen', n_features=10, save_path="./selection.png")

#plot_selection(pod, 'nitrogen', n_features=10, methods='CARS', plot_type='heatmap')

#ivissa.fit(x, n)
#print(ivissa.get_support(indices=True))


