import numpy as np

from auswahl import FeatureDescriptor
from .data_handling import DataHandler
from abc import ABCMeta, abstractmethod


class StabilityScore(metaclass=ABCMeta):

    """
        Base class for all stability scores useable by the benchmarking system

        Parameters
        ----------
        metric_name: str, default=None
            Unique Name of the metric. If no name is provided, the name of the class inheriting from this function
            is used
    """

    def __init__(self, metric_name: str):
        if metric_name is not None:
            self.metric_name = metric_name
        else:
            self.metric_name = type(self).__name__

    def __call__(self, pod: DataHandler):
        self.add_stabilities(pod)

    def add_stabilities(self, pod: DataHandler):
        """Conducts the evaluation of the stability metric across all datasets and methods in the
        :class:`~auswahl.benchmarking.DataHandler` object, which is extended with the results of the
        stability evaluation.

        Parameters
        ----------
        pod: DataHandler
            instance of :class:`~auswahl.benchmarking.DataHandler` containing the results of the benchmarking
            procedure
        """
        for n in pod.feature_descriptors:  # FeatureDescriptor
            for method in pod.methods:
                for dataset in pod.datasets:
                    # retrieve the samples of selected features (list of objects of type Selection)
                    supports = pod.get_selection_data(method=method, n_features=n, dataset=dataset).to_numpy().tolist()
                    supports = np.array([selection.features for selection in supports if selection.is_valid()])

                    stability = self.evaluate_stability(pod.get_meta(dataset), supports, n)

                    if stability is not None:
                        pod.register_stability(method=method,
                                               n_features=n,
                                               dataset=dataset,
                                               metric_name=self.metric_name,
                                               value=stability)

    @abstractmethod
    def evaluate_stability(self, meta_data: dict, selections: np.array, features: FeatureDescriptor) -> float:
        """Conducts the stability evaluation of a set of executions of a selector algorithm on one dataset with
        a specific feature configuration under different data splits and seeds

        Parameters
        ----------
        meta_data: dict
            information about the data set, which might be relevant for stability calculations.
            See :meth:`~auswahl.benchmarking.DataHandler.get_meta` for the contained data
        selections: np.ndarray
            The selected features of the different executions of the selector algorithm as integer indices
            of features. Shape (#executions, #features to select)
        features: FeatureDescriptor
            FeatureDescriptor describing the configuration of features to be selected

        Returns
        -------
        stability: float
        """
        ...


class PairwiseStabilityScore(StabilityScore, metaclass=ABCMeta):

    """
        The class provides the infrastructure for the introduction of new symmetric and pairwise defined
        stability metrics.
    """

    # go
    def _pairwise_scoring(self, meta_data: dict, selections: np.array, features: FeatureDescriptor):
        """The function handles the calculation of a pairwise stability assessment function all executions
        of a selector for a specific dataset and feature configuration

       Parameters
        ----------
        meta_data: dict
            information about the data set, which might be relevant for stability calculations.
            See :meth:`auswahl.DataHandler.get_meta` for the contained data
        selections: np.ndarray
            The selected features of the different executions of the selector algorithm as integer indices
            of features. Shape (#executions, #features to select)
        features: FeatureDescriptor
            FeatureDescriptor describing the configuration of features to be selected

        Returns
        -------
        stability: float
        """

        # evaluate all different pairs (symmetry assumed)
        pairwise_sim = []
        dim0, dim1 = np.triu_indices(selections.shape[0])
        for i in range(dim0.size):
            if dim0[i] != dim1[i]:  # only consider similarity between different pairs of feature sets
                pairwise_sim.append(self.pairwise_sim_func(meta_data,
                                                           set_1=selections[dim0[i]],
                                                           set_2=selections[dim1[i]]))

        if len(pairwise_sim) > 0:
            score = np.mean(np.array(pairwise_sim))
            return score
        return None

    # go
    def evaluate_stability(self, meta_data: dict, selections: np.array, features: FeatureDescriptor):
        return self._pairwise_scoring(meta_data, selections, features)

    # go
    @abstractmethod
    def pairwise_sim_func(self, meta_data: dict, set_1: np.ndarray, set_2: np.ndarray) -> float:
        """Function calculating the stability score for a single pair of selections of features.

        Parameters
        ----------
        meta_data: dict
            Dict containing meta information about the dataset for which the stability metric is evaluated.
            See the documentation of :meth:`~auswahl.benchmarking.DataHandler.get_meta` for the available data.
        set_1: np.nadarray
            array of integer indices of selected features of shape (n_features_to_select,)
        set_2: np.nadarray
            array of integer indices of selected features of shape (n_features_to_select,)

        Returns
        -------
        stability score for the given pair of selections: float

        """
        ...


class DengScore(PairwiseStabilityScore):

    """Wraps the calculation of the selection stability score for randomized selection methods, according to Deng et al. [1]_.
    A detailed overview is provided in the user guide.

    Parameters
    ----------
    metric_name: str, default="deng_score"
            Unique Name of the metric

    References
    ----------
    .. [1] Bai-Chuan Deng, Yong-Huan Yun, Pan Ma, Chen-Chen Li, Da-Bing Ren and Yi-Zeng Liang,
           'A new method for wavelength interval selection that intelligently optimizes the locations, widths
           and combination of intervals',
           Analyst, 6, 1876-1885, 2015.
    """

    def __init__(self, metric_name: str = "deng_score"):
        super().__init__(metric_name)

    def pairwise_sim_func(self, meta_data: dict, set_1: np.ndarray, set_2: np.ndarray) -> float:
        n_wavelengths = meta_data['n_features']
        n = set_1.size
        e = n ** 2 / n_wavelengths
        return (np.intersect1d(set_1, set_2).size - e) / (n - e)


class ZucknickScore(PairwiseStabilityScore):
    """Wraps the calculation of the stability score according to Zucknick et al. [1]_. The stability score features a
    correlation-adjusting mechanism assessing stability not only with respect to
    set theoretical stabilities, but also according to the correlation between the features selected in different runs.
    A detailed overview is provided in the userguide.

    Parameters
    ----------
    correlation_threshold: float, default=0.8
        Parameter of the calculation of stability according to Zucknick et al. [1]_ . The parameter determines
        the minimum required correlation between two features to be considered similar.
    metric_name: str, default="zucknick_score"

    References
    ----------
    .. [1] Zucknick, M., Richardson, S., Stronach, E.A.: Comparing the characteristics of
           gene expression profiles derived by univariate and multivariate classification methods.
           Stat. Appl. Genet. Molecular Biol. 7(1), 7 (2008)
    """

    def __init__(self, correlation_threshold: float = 0.8, metric_name: str = "zucknick_score"):
        super().__init__(metric_name)

        if 0 <= correlation_threshold <= 1:
            self.correlation_threshold = correlation_threshold
        else:
            raise ValueError(f'Argument correlation_threshold is required to be in [0, 1]')

    def _thresholded_correlation(self, spectra, support_1: np.array, support_2: np.array):
        set_diff = np.setdiff1d(support_2, support_1)
        if set_diff.size == 0:
            return 0
        diff_features = np.transpose(spectra[:, set_diff])  # features x observations
        sup1_features = np.transpose(spectra[:, support_1])
        correlation = np.abs(np.corrcoef(sup1_features, diff_features))
        correlation = correlation * (correlation >= self.correlation_threshold)
        return (1 / support_2.size) * np.sum(correlation[:support_1.size, support_1.size:])

    def pairwise_sim_func(self, meta_data: dict, set_1: np.ndarray, set_2: np.ndarray) -> float:
        n = set_1.size
        spectra = meta_data['x']
        intersection_size = np.intersect1d(set_1, set_2).size
        union_size = 2 * n - intersection_size
        c_12 = self._thresholded_correlation(spectra, set_1, set_2)
        c_21 = self._thresholded_correlation(spectra, set_2, set_1)
        return (intersection_size + c_12 + c_21) / union_size
