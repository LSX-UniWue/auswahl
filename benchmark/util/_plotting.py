import numpy as np
from typing import List
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def plot_score_stability_box(scores: List[np.array],
                             stabilities: List[float],
                             method_names: List[str],
                             score_metric_name: str):
    """
            TODO
    """
    # TODO: check dimensinoality
    # TODO: color strategies
    colors = ['cyan', 'red', 'green']
    entities = ['boxprops', 'medianprops', 'flierprops', 'capprops', 'whiskerprops']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    legend_handles = []
    for i, score in enumerate(scores):
        kwargs = dict()
        for entity in entities:
            kwargs[entity] = dict(color=colors[i])
        ax.boxplot(score, positions=[stabilities[i]], whis=(0, 100), **kwargs)
        legend_handles.append(mpatches.Patch(color=colors[i], label=method_names[i]))

    ax.set_title("Score-Stability plot")
    ax.set_xlabel("stability score")
    ax.set_ylabel(f'{score_metric_name}')
    ax.legend(handles=legend_handles)
    plt.show()

# testing
if __name__ == "__main__":
    np.random.seed(19680801)
    data = [np.random.normal(0, std, 100) for std in range(8, 10)]
    plot_score_stability_box(data, [2.5,3.6], ['CARS', 'IPLS'], 'rmsecv')