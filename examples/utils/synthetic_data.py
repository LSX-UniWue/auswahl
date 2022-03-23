import numpy as np
from scipy.stats import multivariate_normal
from _plotting import plot_spectra

from sklearn.utils import check_random_state

class SyntheticGenerator:

    def __init__(self, random_state):
        self.random_state = random_state

    def _generate_seed(self, n_wavelengths: int, random_state: np.random.RandomState):
        acc = (2 * random_state.rand(n_wavelengths - 2) - 1)
        return acc, 0.005 * random_state.rand(), random_state.rand(), random_state.rand()

    def _generate_samples(self, n_samples, acc, velo, start, end, random_state):

        acc = acc * (0.2 * random_state.rand(n_samples, acc.shape[0]) + 0.8)
        acc_offset = (end - start) - (velo * (acc.shape[1] + 1))
        acc_multiples = np.expand_dims(np.arange(acc.shape[1], 0, -1), axis=0)
        acc_norm = acc_offset / np.sum(acc*acc_multiples, axis=1)
        acc = acc * np.expand_dims(acc_norm, axis=1)

        # TODO: switch to sampling with covariances to produce randomly selected correlation between wavelengths
        # cov = np.eye(seed.shape[0] - 2)
        # sample = np.random.multivariate_normal(np.zeros((seed.shape[0]-2)), cov, size=n, check_valid='warn')
        # sample = np.array(sample).reshape((n, seed.shape[0]-2))

        # probabilistic_mask = 0.2 * np.abs(sample) + 0.8
        # acc = acc * probabilistic_mask
        #TODO: parallelize
        velocity = np.zeros((n_samples, acc.shape[1] + 1))
        velocity[:, 0] = velo * np.ones((n_samples,))
        for i in range(acc.shape[1]):
            velocity[:, i + 1] = velocity[:, i] + acc[:, i]

        trajectory = np.zeros((n_samples, acc.shape[1] + 2))
        trajectory[:, 0] = start + 0.06 * np.random.rand(trajectory.shape[0])
        for i in range(velocity.shape[1]):
            trajectory[:, i + 1] = trajectory[:, i] + velocity[:, i]

        # renormalize spectra
        trajectory = trajectory - np.min(trajectory) + random_state.rand() * 0.25
        trajectory = trajectory * ((random_state.rand() * 0.3 + 0.6) / np.max(trajectory))

        #mask = np.all(trajectory >= 0, axis=1)
        #trajectory = np.compress(mask, trajectory, axis=0)
        plot_spectra(np.arange(trajectory.shape[1]),
                     trajectory,
                     "Reflectance",
                     "Wavenumber",
                     "synthetic data",
                     "./all_synthetic.png")

    def generate(self, n_samples, n_wavelengths):

        random_state = check_random_state(self.random_state)
        acc, velo, start, end = self._generate_seed(n_wavelengths, random_state)

        samples = self._generate_samples(n_samples, acc, velo, start, end, random_state)

        # TODO: add some noise, calculate a target quantity on the noised data
        # TODO: return targets and no-noised samples
        # TODO: extend interface for number of wavelengths/intervals involved in the calculation of the target


gen = SyntheticGenerator(np.random.randint(0, 100000))

gen.generate(20, 700)









