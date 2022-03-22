
from typing import List, Literal
from matplotlib import pyplot as plt

import numpy as np


def plot_spectra(wavelengths: np.array,
                 measurements: np.array,
                 quantity_name: str = 'Absorbance',
                 x_name: str = 'Wavenumber',
                 plot_name: str = 'Spectra',
                 save_path: str = './spectra.png',
                 highlight_ranges: List[tuple] = None,
                 highlight_color: Literal['red', 'green', 'blue'] = 'red',
                 highlight_opacity: float = 0.3):
    fig, ax = plt.subplots()
    ticks = np.arange(wavelengths[0], wavelengths[-1] + 9, wavelengths.shape[0] // 10)

    ax.get_xaxis().set_ticks(ticks)
    ax.set_xlabel(x_name)
    ax.set_xticklabels(ticks)

    ax.set_ylabel(quantity_name)

    if len(measurements.shape) == 1:
        measurements = np.expand_dims(measurements, axis=0)

    plot_args = []
    for i in range(measurements.shape[0]):
        plot_args += [wavelengths, measurements[i, :]]

    # highlighting
    if highlight_ranges is not None:
        for start, end in highlight_ranges:
            ax.axvspan(start, end, color=highlight_color, alpha=highlight_opacity)

    ax.plot(*plot_args)
    plt.title(plot_name)
    plt.savefig(save_path)
