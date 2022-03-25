
import numpy as np
from auswahl import IPLS, VIP, MCUVE, CARS
from utils import SyntheticGenerator, plot_spectra

seed = 26902
generator = SyntheticGenerator(seed)
spectra, targets, gold = generator.generate_interval_dataset(50, 700, 5, 30)

# snv
spectra_prep = (spectra - np.mean(spectra, axis=1, keepdims=True)) / np.std(spectra, axis=1, keepdims=True)

cars = CARS(n_features_to_select=10, random_state=12345)
cars.fit(spectra_prep, targets)

selected = np.nonzero(cars.support_)[0]

plot_spectra(np.arange(spectra.shape[1]),
             spectra,
             plot_name="Selected interval",
             save_path="selected_intervals.png",
             highlight_ranges=[(i, i + 1) for i in selected])

plot_spectra(np.arange(spectra.shape[1]),
             spectra,
             plot_name="Ground truth interval",
             save_path="./ground_truth_intervals.png",
             highlight_ranges=[(gold[i, 0], gold[i, -1]) for i in range(gold.shape[0])])



