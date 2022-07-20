import numpy as np

from .data_handling import BenchmarkPOD


# go
def _pairwise_scoring(pod: BenchmarkPOD, pairwise_sim_function, metric_name: str):
    r = pod.n_runs
    for n in pod.feature_descriptors:  # FeatureDescriptor
        for method in pod.methods:
            for dataset in pod.datasets:
                # retrieve the samples of selected features (list of objects of type Selection)
                supports = pod.get_selection_data(method=method, n_features=n, dataset=dataset).to_numpy().tolist()
                supports = np.array([selection.selected_features for selection in supports])

                pairwise_sim = []
                dim0, dim1 = np.triu_indices(r)
                for i in range(dim0.size):
                    if dim0[i] != dim1[i]:  # only consider similarity between different pairs of feature sets
                        pairwise_sim.append(pairwise_sim_function(pod,
                                                                  support_1=supports[dim0[i]],
                                                                  support_2=supports[dim1[i]],
                                                                  n_features=len(n),  # number of features selected
                                                                  method=method,
                                                                  dataset=dataset))
                score = np.sum(np.array(pairwise_sim)) * (2 / (r * (r - 1)))

                pod.register_stability(method=method,
                                       n_features=n,
                                       dataset=dataset,
                                       metric_name=metric_name,
                                       value=score)


# go
def _deng_stability_score(pod: BenchmarkPOD, support_1: np.array, support_2: np.array, **kwargs):
    n_wavelengths = pod.get_meta(kwargs['dataset'])[2][1]
    n = kwargs['n_features']
    e = n ** 2 / n_wavelengths
    return (np.intersect1d(support_1, support_2).size - e) / (n - e)


# go
def deng_score(pod: BenchmarkPOD):
    """
            Calculates the selection stability score for randomized selection methods, according to Deng et al. [1]_.

            Parameters
            ----------
            pod : BenchmarkPOD
                data container produced by benchmarking

            Returns
            -------
                Extends the passed BenchmarkPOD with the stability scores calculated according to Deng et al. [1]_

            References
            ----------
            .. [1] Bai-Chuan Deng, Yong-Huan Yun, Pan Ma, Chen-Chen Li, Da-Bing Ren and Yi-Zeng Liang,
                   'A new method for wavelength interval selection that intelligently optimizes the locations, widths
                   and combination of intervals',
                   Analyst, 6, 1876-1885, 2015.

    """
    _pairwise_scoring(pod, _deng_stability_score, 'deng_score')


def _thresholded_correlation(spectra, support_1: np.array, support_2: np.array, threshold: float):
    set_diff = np.setdiff1d(support_2, support_1)
    if set_diff.size == 0:
        return 0
    diff_features = np.transpose(spectra[:, set_diff])  # features x observations
    sup1_features = np.transpose(spectra[:, support_1])
    correlation = np.abs(np.corrcoef(sup1_features, diff_features))
    correlation = correlation * (correlation >= threshold)
    return (1/support_2.size) * np.sum(correlation[:support_1.size, support_1.size:])


def _zucknick_stability_score(pod: BenchmarkPOD, support_1: np.array, support_2: np.array, **kwargs):
    n = kwargs['n_features']
    spectra = pod.get_meta(kwargs['dataset'])[0]
    intersection_size = np.intersect1d(support_1, support_2).size
    union_size = 2*n - intersection_size
    c_12 = _thresholded_correlation(spectra, support_1, support_2, 0.8)
    c_21 = _thresholded_correlation(spectra, support_2, support_1, 0.8)
    return (intersection_size + c_12 + c_21) / union_size


def zucknick_score(pod: BenchmarkPOD):
    """
            Calculates the stability score according to Zucknick et al. _[1]

        Parameters
        ----------
        pod: BenchmarkPOD
            BenchmarkPOD object containing benchmarking data

        References
        ----------
        _[1] Zucknick, M., Richardson, S., Stronach, E.A.: Comparing the characteristics of
             gene expression profiles derived by univariate and multivariate classification methods.
             Stat. Appl. Genet. Molecular Biol. 7(1), 7 (2008)
    """
    _pairwise_scoring(pod, _zucknick_stability_score, 'zucknick_score')

