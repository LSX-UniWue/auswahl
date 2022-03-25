
import numpy as np
from sklearn.utils import check_random_state

class SyntheticGenerator:

    def __init__(self, random_state):
        self.random_state = random_state

    def _generate_samples(self, n_samples, n_wavelengths, random_state):

        # acceleration seed values drawn from [-1,1]
        acc = 2 * random_state.rand(n_wavelengths - 2) - 1
        # initial velocity
        init_velo = 0.005 * random_state.rand()
        # starting and ending value of the spectra
        start = random_state.rand()
        end = random_state.rand()

        # create a mask of random effects acting on the acceleration seed
        acc = acc * (0.2 * random_state.rand(n_samples, acc.shape[0]) + 0.8)
        # force second integral of acceleration given initial velocity to span the spectral interval (end - start)
        acc_offset = (end - start) - (init_velo * (acc.shape[1] + 1))
        acc_multiples = np.expand_dims(np.arange(acc.shape[1], 0, -1), axis=0)
        acc_norm = acc_offset / np.sum(acc*acc_multiples, axis=1)
        acc = acc * np.expand_dims(acc_norm, axis=1)

        # integrate velocities
        velocity = np.zeros((n_samples, acc.shape[1] + 1))
        velocity[:, 0] = init_velo * np.ones((n_samples,))
        for i in range(acc.shape[1]):
            velocity[:, i + 1] = velocity[:, i] + acc[:, i]

        # integrate trajectory, with random translational displacement
        trajectory = np.zeros((n_samples, acc.shape[1] + 2))
        trajectory[:, 0] = start + 0.06 * np.random.rand(trajectory.shape[0])
        for i in range(velocity.shape[1]):
            trajectory[:, i + 1] = trajectory[:, i] + velocity[:, i]

        # re-normalize spectra into [0,1]
        trajectory = trajectory - np.min(trajectory) + random_state.rand() * 0.25
        trajectory = trajectory * ((random_state.rand() * 0.3 + 0.6) / np.max(trajectory))

        # drop exotic spectra with high range (this can reduce the actual number of spectra churned out)
        spectral_range_median = np.median(trajectory, axis=0)
        spectral_deviations_median = np.median(np.abs(trajectory - spectral_range_median), axis=0)
        well_behaved_spectra = np.all(np.abs(trajectory - spectral_range_median) <= 1.5 * spectral_deviations_median,
                                      axis=1)

        trajectory = np.compress(well_behaved_spectra, trajectory, axis=0)

        return trajectory

    def _preprocess(self, samples, random_state):
        # SNV
        samples = (samples - np.mean(samples, axis=1, keepdims=True)) / np.std(samples, axis=1, keepdims=True)
        # Add noise
        samples = samples + random_state.normal(0, 0.005, samples.shape)
        return samples

    def _scramble_points(self, samples, n_targets, random_state):
        samples = self._preprocess(samples, random_state)
        points = random_state.choice(np.arange(0, samples.shape[1]),
                                     n_targets,
                                     replace=False)
        weights = random_state.choice([-1, 1], 1) * random_state.normal(1, 1, n_targets)

        targets = np.sum(samples[:, points] * weights, axis=1)
        return targets, points

    def _scramble_intervals(self, samples, n_intervals, interval_width, random_state):
        samples = self._preprocess(samples, random_state)
        # place non-overlapping intervals
        interval_starts = np.arange(0, n_intervals) * interval_width
        for i in range(n_intervals):
            interval_starts[i:] = interval_starts[i:] + random_state.randint(0, 0.5 * (samples.shape[1]
                                                                             - interval_starts[-1]
                                                                             - interval_width))

        selected_wavelengths = np.repeat(interval_starts, interval_width) + np.tile(np.arange(0, interval_width),
                                                                                    n_intervals)

        weights = random_state.choice([-1, 1], 1) * random_state.normal(4, 1, n_intervals * interval_width)
        targets = np.sum(samples[:, selected_wavelengths] * weights, axis=1)
        return targets, selected_wavelengths.reshape((n_intervals, -1))

    def generate_interval_dataset(self, n_samples, n_wavelengths, n_intervals, interval_width):

        random_state = check_random_state(self.random_state)

        samples = self._generate_samples(n_samples, n_wavelengths, random_state)
        targets, selected_wavelengths = self._scramble_intervals(samples, n_intervals, interval_width, random_state)
        return samples, targets, selected_wavelengths

    def generate_point_dataset(self, n_samples, n_wavelengths, n_targets):
        random_state = check_random_state(self.random_state)

        samples = self._generate_samples(n_samples, n_wavelengths, random_state)
        targets, selected_wavelengths = self._scramble_points(samples, n_targets, random_state)
        return samples, targets, selected_wavelengths











