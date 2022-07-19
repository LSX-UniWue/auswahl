
import warnings
import pandas as pd

import matplotlib.patches as mpatches
import numpy as np

from typing import List, Union, Literal, Tuple

from matplotlib import pyplot as plt

from .data_handling import BenchmarkPOD
from ..._base import FeatureDescriptor


def _to_string(items: list):
    return [str(i) for i in items]


# go
def _check_specified_or_singleton(pool, argument, identifier):
    if argument is None:
        if len(pool) > 1:
            raise ValueError(f'{identifier} is ambiguous. Specify a {identifier}.')
        elif len(pool) == 0:
            raise ValueError(f'No {identifier} specified during configuration of the benchmarking.')
        return pool[0]
    return argument #argument if isinstance(argument, str) else argument.__name__


# go
def _arrange_boxes(pod, n_features, methods):
    x_coords = []
    ticks = np.arange(len(n_features) if n_features is not None else len(pod.feature_descriptors)) + 1  #start with 1
    n_methods = len(methods if methods is not None else pod.methods)
    if len(methods) > 1:
        for i in range(n_methods):
            x_coords.append((-0.15 + ticks + (0.3 / (n_methods - 1)) * i).tolist())
    else:
        x_coords = [ticks]
    return x_coords, ticks


# go
def _box_plot(title: str,
              x_label: str,
              y_label: str,
              y_data: List[List[float]],
              x_data: List[float],
              legend: List[str],
              tick_labels: List[List[Union[float, int]]] = None,
              ticks: List[Union[int, float]] = None,
              save_path: str = None):

    colors = plt.cm.get_cmap('Accent', len(y_data) + 1)
    entities = ['boxprops', 'medianprops', 'flierprops', 'capprops', 'whiskerprops']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    legend_handles = []

    # adapt box width to the offsetting and the number of methods if a custom ticking is used
    if tick_labels is not None:
        box_width = (0.3 * 0.9) / len(y_data)
    else:
        box_width = 0.05

    if legend is None:
        plotting_kwargs = dict()
        for entity in entities:
            plotting_kwargs[entity] = dict(color='k')

    for i, data in enumerate(y_data):
        if legend is not None:
            plotting_kwargs = dict()
            for entity in entities:
                plotting_kwargs[entity] = dict(color=colors(i))

        ax.boxplot(data, positions=x_data[i], whis=(0, 100), widths=box_width, manage_ticks=False, **plotting_kwargs)

        if legend is not None:
            legend_handles.append(mpatches.Patch(color=colors(i), label=legend[i]))

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(axis='y')

    if ticks is not None:  # apply custom ticking and labelling
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ax.set_xticklabels(tick_labels)
            ax.set_xticks(ticks)

    if legend is not None:
        ax.legend(handles=legend_handles)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


# go
def _errorbar_plot(title: str,
                   x_label: str,
                   y_label: str,
                   y_data: np.array,
                   y_max: np.array,
                   y_min: np.array,
                   x_data: List[List[Union[float, int]]],
                   tick_labels: List[List[Union[float, int]]],
                   ticks: List[Union[int, float]],
                   legend: List[str],
                   plot_lines: bool = True,
                   save_path: str = None):

    colors = plt.cm.get_cmap('Accent', y_data.shape[0] + 1)
    markers = [c for c in ".ov^<>12348sp*hH+xDd|"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    legend_handles = []

    #calculate errors:
    y_max = (y_max - y_data).tolist()
    y_min = (y_data - y_min).tolist()
    y_data = y_data.tolist()

    for i, y in enumerate(y_data):
        ax.errorbar(x_data[i] if len(x_data) > 1 else x_data[0],
                    y,
                    yerr=[y_min[i], y_max[i]],
                    color=colors(i),
                    marker=markers[i],
                    linestyle='dotted' if plot_lines else 'none')
        legend_handles.append(mpatches.Patch(color=colors(i), label=legend[i]))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticklabels(tick_labels)
        ax.set_xticks(ticks)
        ax.legend(handles=legend_handles)
        ax.grid(axis='y')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def _line_plot(title: str,
               x_label: str,
               y_label: str,
               y_data: List[List[float]],
               x_data: List[Union[int, Tuple[int, int]]],
               legend: List[str],
               save_path: str = None):

    colors = plt.cm.get_cmap('Accent', len(y_data) + 1)
    markers = [c for c in ".ov^<>12348sp*hH+xDd|"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    legend_handles = []

    positions = np.arange(len(x_data))
    for i, y_data in enumerate(y_data):
        ax.errorbar(positions,
                    y_data,
                    color=colors(i),
                    marker=markers[i])
        legend_handles.append(mpatches.Patch(color=colors(i), label=legend[i]))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks(positions)
        ax.set_xticklabels(x_data)
        ax.legend(handles=legend_handles)
        ax.grid(axis='y')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


# go -> confirmed
def plot_score_vs_stability(pod: BenchmarkPOD,
                            n_features: Union[int, Tuple[int]] = None,
                            dataset: str = None,
                            stability_metric: str = None,
                            regression_metric: str = None,
                            methods: Union[str, List[str]] = None,
                            save_path: str = None):
    """
        Plotting a boxplot for the benchmarked methods displaying
            * the mean regression score on the y-axis
                * mean regression value
                * (25,75) IQR as box
                * (0, 100) range as whiskers
            * the stability score on the x-axis

        Parameters
        ----------

        pod : BenchmarkPOD
            data container produced by benchmarking

        dataset: str
            dataset for which the data is to be plotted

        n_features: int or tuple of int
            number of features, which were to be selected by the algorithms

        stability_metric : str
            identifier of the stability metric to be plotted in the pod

        regression_metric : str
            identifier of the regression metric to be plotted in the pod

        methods : str or list of str, default=None
            identifers of methods for which the data is to be plotted. If None, all available methods are plotted

        save_path: str, default=None
            path at which the plot is stored. If None, the plot is just displayed
    """

    dataset = _check_specified_or_singleton(pod.datasets, dataset, identifier='dataset')
    n_features = _check_specified_or_singleton(pod.feature_descriptors, n_features, identifier='n_features')
    regression_metric = _check_specified_or_singleton(pod.reg_metrics, regression_metric, identifier='regression metric')
    stability_metric = _check_specified_or_singleton(pod.stab_metrics, stability_metric, identifier='stability metric')

    reg_data = pod.get_regression_data(dataset=dataset,
                                       method=methods,
                                       n_features=n_features,
                                       reg_metric=regression_metric).to_numpy().tolist()

    stab_data = pod.get_stability_data(dataset=dataset,
                                       method=methods,
                                       n_features=n_features,
                                       stab_metric=stability_metric).to_numpy().tolist()

    _box_plot("Regression-Stability-Plot",
              stability_metric,
              regression_metric,
              reg_data,
              stab_data,
              pod.methods,
              save_path=save_path)


# go -> confirmed
def plot_exec_time(pod: BenchmarkPOD,
                   dataset: str = None,
                   methods: Union[str, List[str]] = None,
                   n_features: List[Union[int, Tuple[int]]] = None,
                   item: Literal['mean', 'median'] = 'mean',
                   save_path: str = None):

    """
        Plots execution times of selectors across different number of features to be selected

    Parameters
    ----------
    pod: BenchmarkPOD
        BenchmarkPOD object containing the benchmarking data
    dataset: str, default=None
        identifier of the dataset of which to plot the execution time. If there is data for only one dataset
        in the BenchmarkPOD object, the argument does not have to be specified
    methods: str or list of str, default=None
        identifiers of methods for which to plot the execution time. If None, all available methods are used.
    n_features: list of integers or of tuples of integers, default=None
        identifiers of the number of features or the configuration of intervals for which the execution time is to be plotted.
        If None, all available feature descriptors are used.
    item: Literal['mean', 'median'], default='mean'
        specifies whether the mean or median is displayed in the plot
    save_path: str
        path at which the plot has to be saved

    """

    dataset = _check_specified_or_singleton(pod.datasets, dataset, identifier='dataset')

    if n_features is not None and not isinstance(n_features, list):
        n_features = [n_features]

    if methods is None:
        methods = pod.methods

    exec_times = pod.get_measurement_data(dataset=dataset,
                                          method=methods,
                                          n_features=n_features)

    grouped = exec_times.groupby(axis=1, level=['dataset', 'n_features'])

    exec_mins = grouped.min().to_numpy()
    exec_max = grouped.max().to_numpy()

    if item == 'mean':
        exec_times = grouped.mean().to_numpy()
    elif item == 'median':
        exec_times = grouped.median().to_numpy()
    else:
        raise ValueError("f'Unknown item {item}. Use median or mean'")

    x_coords, ticks = _arrange_boxes(pod, n_features, methods)

    _errorbar_plot(f'Execution time: {item} and ranges',
                   "n_features",
                   "Execution time [s]",
                   exec_times,
                   exec_max,
                   exec_mins,
                   x_coords,
                   n_features if n_features is not None else _to_string(pod.feature_descriptors),
                   ticks,
                   methods if methods is not None else pod.methods,
                   plot_lines=False,
                   save_path=save_path)


# go -> confirmed
def _plot_score_box(pod: BenchmarkPOD,
                    dataset: str,
                    regression_metric,
                    methods: Union[str, List[str]],
                    n_features: List[Union[int, Tuple[int, int]]],
                    save_path: str = None):

    regression_metric = _check_specified_or_singleton(pod.reg_metrics, regression_metric,
                                                      identifier='regression metric')
    dataset = _check_specified_or_singleton(pod.datasets, dataset, identifier='dataset')

    if methods is None:
        methods = pod.methods

    if n_features is not None and not isinstance(n_features, list):
        n_features = [n_features]

    n_features = [FeatureDescriptor(feature, pod.resolve_tuples) for feature in n_features]

    reg_scores = pod.get_regression_data(method=methods,
                                         n_features=n_features,
                                         dataset=dataset,
                                         reg_metric=regression_metric).to_numpy()
    reg_scores = np.reshape(reg_scores, (len(methods), -1, pod.n_runs)).tolist()  # shape: methods x n_features x n_runs

    # calculate offset x coordinates
    x_coords, ticks = _arrange_boxes(pod, n_features, methods)

    _box_plot(f'Regression performance box plot on dataset {dataset}',
              "n_features",
              regression_metric,
              reg_scores,
              x_coords,
              methods if methods is not None else pod.methods,
              n_features if _to_string(n_features) is not None else _to_string(pod.feature_descriptors),
              ticks,
              save_path=save_path)


# TODO: adapt
def _plot_score_bar(pod: BenchmarkPOD,
                    dataset: str = None,
                    regression_metric: str = None,
                    methods: Union[str, List[str]] = None,
                    n_features: List[Union[int, Tuple[int, int]]] = None,
                    item: Literal['mean', 'median'] = 'mean',
                    save_path: str = None):

    regression_metric = _check_specified_or_singleton(pod.reg_metrics, regression_metric,
                                                      identifier='regression metric')
    dataset = _check_specified_or_singleton(pod.datasets, dataset, identifier='dataset')

    if n_features is not None and not isinstance(n_features, list):
        n_features = [n_features]

    reg_scores = pod.get_regression_data(method=methods,
                                         n_features=n_features,
                                         dataset=dataset,
                                         reg_metric=regression_metric,
                                         item=item).to_numpy()

    reg_mins = pod.get_regression_data(method=methods,
                                       dataset=dataset,
                                       n_features=n_features,
                                       reg_metric=regression_metric,
                                       item='min').to_numpy()

    reg_max = pod.get_regression_data(method=methods,
                                      dataset=dataset,
                                      n_features=n_features,
                                      reg_metric=regression_metric,
                                      item='max').to_numpy()

    # calculate offset x coordinates
    x_coords, ticks = _arrange_boxes(pod, n_features, methods)

    _errorbar_plot(f'Regression performance: {item} and range on dataset {dataset}',
                   "n_features",
                   regression_metric,
                   reg_scores,
                   reg_max,
                   reg_mins,
                   x_coords,
                   n_features if n_features is not None else pod.n_features,
                   ticks,
                   methods if methods is not None else pod.methods,
                   plot_lines=False,
                   save_path=save_path
                   )


# go -> confirmed
def plot_score(pod: BenchmarkPOD,
               dataset: str = None,
               regression_metric: str = None,
               methods: Union[str, List[str]] = None,
               n_features: List[Union[int, Tuple[int, int]]] = None,
               item: Literal['mean', 'median'] = 'mean',
               plot_type: Literal['box', 'bar'] = 'box',
               save_path: str = None):

    """
        Plot regression scores of selectors across different number of features to be selected as box or bar plot

    Parameters
    ----------
    pod: BenchmarkPOD
        BenchmarkPOD object containing the benchmarking data
    dataset: str, default=None
        identifier of the dataset of which to plot the execution time. If there is data for only one dataset
        in the BenchmarkPOD object, the argument does not have to be specified
    methods: str or list of str, default=None
        identifiers of methods for which to plot the execution time. If None, all available methods are used.
    n_features: list of integers or of tuples of integers, default=None
        identifiers of the number of features or the configuration of intervals for which the execution time is to be plotted.
        If None, all available feature descriptors are used.
    item: Literal['mean', 'median'], default='mean'
        specifies whether the mean or median is displayed in the plot
    plot_type: Literal['box', 'bar'], default='box'
        specifies the requested plot type
    save_path: str, default=None
        path at which the plot has to be saved. If None, the plot is only displayed, not saved.

    """
    if plot_type == 'box':
        _plot_score_box(pod, dataset, regression_metric, methods, n_features, save_path)
    elif plot_type == 'bar':
        _plot_score_bar(pod, dataset, regression_metric, methods, n_features, item, save_path)
    else:
        raise ValueError(f'Unknown plot type {plot_type}.')


# go -> confirmed
def plot_stability(pod: BenchmarkPOD,
                   dataset: str = None,
                   stability_metric: str = None,
                   methods: Union[str, List[str]] = None,
                   save_path: str = None):

    """
        Plots the stability score of methods for a given metric across the number of features to be selected

        Parameters
        ----------
        pod: BenchmarkPOD
            BenchmarkPOD object containing the benchmarking data
        dataset: str
            dataset identifier
        stability_metric: str
            stability metric used for plotting
        methods: Union[str, List[str]], default=None
            method identifier or list of method identifiers for methods to be plotted. If None, all available methods
            are plotted
        save_path: str, default=None
            path on which to store the plot. If None, the plot is simply displayed

    """
    dataset = _check_specified_or_singleton(pod.datasets, dataset, identifier='dataset')
    stability_metric = _check_specified_or_singleton(pod.stab_metrics, stability_metric, identifier='stability metric')

    y_data = pod.get_stability_data(method=methods, dataset=dataset, stab_metric=stability_metric).to_numpy().tolist()
    x_data = _to_string(pod.feature_descriptors)

    _line_plot(f'Stability across n_features to select: Dataset {dataset}',
               "n_features",
               stability_metric,
               y_data,
               x_data,
               pod.methods,
               save_path)


# go
def _plot_selection_bar(pod: BenchmarkPOD,
                        dataset: str,
                        n_features: Union[int, Tuple[int]],
                        methods: Union[str, List[str]] = None,
                        save_path: str = None):

    if methods is None:
        methods = pod.methods

    colors = plt.cm.get_cmap('Accent', len(methods) + 1)

    fig = plt.figure()
    gs = fig.add_gridspec(len(methods), hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    fig.suptitle(f'Selection probability P on dataset {dataset} for {n_features} features.')

    selections = pod.get_selection_data(dataset=dataset, method=methods, n_features=n_features).to_numpy().tolist()
    selections = pd.DataFrame([sum([s.selected_features for s in sel], []) for sel in selections])

    n_wavelengths = pod.get_meta(dataset)[2][1]  # TODO: improve this interface

    if len(methods) == 1:
        axs = [axs]

    for i in range(len(methods)):

        unique_counts = selections.iloc[i].value_counts()
        bar_heights = np.zeros((n_wavelengths,))
        bar_heights[unique_counts.index.to_numpy().astype('int')] = unique_counts.to_numpy()

        axs[i].bar(np.arange(n_wavelengths), bar_heights / pod.n_runs, color=colors(i))
        if i % 2 == 0:  # distribute y-axis ticks between left and right-hand side
            axs[i].yaxis.tick_right()
        else:
            axs[i].yaxis.set_label_position("right")

        axs[i].set_xlabel("wavelength")
        axs[i].set_ylabel("P")

        axs[i].legend(handles=[mpatches.Patch(color=colors(i), label=methods[i])])

    # Hide x labels and tick labels for all but the bottom plot.
    for ax in axs:
        ax.label_outer()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


# TODO: probably to be discarded
def _plot_selection_heatmap(pod: BenchmarkPOD,
                            n_features: int,
                            dataset: str = None,
                            methods: Union[str, List[str]] = None,
                            save_path: str = None):

    fig, ax = plt.subplots()
    ax.set_title(f'Displaying selection probability heatmap on dataset {dataset} for {n_features} features to be selected')

    selections = pod.get_selection_data(dataset=dataset, method=methods, n_features=n_features)
    n_wavelengths = pod.get_meta(dataset)[2][1]

    selection_prob = np.zeros((len(methods), n_wavelengths))
    for i in range(len(methods)):
        unique_counts = selections.iloc[i].value_counts()
        selection_prob[i, unique_counts.index.to_numpy().astype('int')] = unique_counts.to_numpy() / pod.n_runs

    #build heatmap
    print(selection_prob)
    ax.imshow(selection_prob, cmap='viridis')

    #add annotations
    ax.set_xlabel('wavelength')
    ax.set_ylabel('method')
    ax.set_yticks(np.arange(len(methods)), labels=methods)
    ax.set_xlabel(np.arange(n_wavelengths))

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


# go -> confirmed
def plot_selection(pod: BenchmarkPOD,
                   n_features: Union[int, Tuple[int, int]],
                   dataset: str = None,
                   methods: Union[str, List[str]] = None,
                   plot_type: Literal['heatmap', 'bar'] = 'bar',
                   save_path: str = None):

    """

        Plots the selection probabililty for features of different selectors

     Parameters
     ----------
    pod: BenchmarkPOD
        BenchmarkPOD object containing the benchmarking data
    dataset: str
        dataset identifier
    stability_metric: str
        stability metric used for plotting
    methods: Union[str, List[str]], default=None
        method identifier or list of method identifiers for methods to be plotted. If None, all available methods
        are plotted
    plot_type: Literal['heatmap', 'bar'], default='bar'
        plot type displayed
    save_path: str, default=None
        path on which to store the plot. If None, the plot is simply displayed

    """
    dataset = _check_specified_or_singleton(pod.datasets, dataset, identifier='dataset')

    if methods is None:
        methods = pod.methods

    if plot_type == 'bar':
        _plot_selection_bar(pod, dataset, n_features, methods, save_path)
    elif plot_type == 'heatmap':
        _plot_selection_heatmap(pod, dataset, n_features, methods, save_path)
    else:
        raise ValueError(f'Unknown plot_type passed to function plot_selection: {plot_type}.'
                         'Use one of {heatmap, bar}')
