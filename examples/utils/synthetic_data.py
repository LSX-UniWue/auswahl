import numpy as np
from scipy.stats import multivariate_normal
from _plotting import plot_spectra

class SyntheticGenerator:

    def __init__(self, random_state):
        ...

    def _generate_seed(self, n_wavelengths):
        """
            TODO: Simply generate a smooth something
        """
        return None

    def _generate_samples(self, n_samples, seed):

        velo = np.diff(seed, 1)
        acc = np.diff(velo, 1)

        n = 20 #  TODO: the generation process is probabilistic as of now, fix that with n_samples

        #proof of concept
        acc = acc * (0.2 * np.random.rand(n, acc.shape[0]) + 0.8)
        acc_offset = (seed[-1] - seed[0]) - (velo[0] * (seed.shape[0] - 1))
        acc_multiples = np.expand_dims(np.arange(seed.shape[0]-2, 0, -1), axis=0)
        acc_norm = acc_offset / np.sum(acc*acc_multiples, axis=1)
        acc = acc * np.expand_dims(acc_norm, axis=1)

        # TODO: switch to sampling with covariances to produce randomly selected correlation between wavelengths
        # cov = np.eye(seed.shape[0] - 2)
        # sample = np.random.multivariate_normal(np.zeros((seed.shape[0]-2)), cov, size=n, check_valid='warn')
        # sample = np.array(sample).reshape((n, seed.shape[0]-2))

        # probabilistic_mask = 0.2 * np.abs(sample) + 0.8
        # acc = acc * probabilistic_mask

        #TODO: parallelize
        velocity = np.tile(velo, (n, 1))
        for i in range(acc.shape[1]):
            velocity[:, i + 1] = velocity[:, i] + acc[:, i]

        trajectory = np.tile(seed, (n, 1))
        trajectory[:, 0] = trajectory[:, 0] + 0.06 * np.random.rand(trajectory.shape[0])
        for i in range(velocity.shape[1]):
            trajectory[:, i + 1] = trajectory[:, i] + velocity[:, i]

        mask = np.all(trajectory >= 0, axis=1)
        trajectory = np.compress(mask, trajectory, axis=0)

        return trajectory

    def generate(self, n_samples, n_wavelengths):

        seed = self._generate_seed(n_wavelengths)

        samples = self._generate_samples(n_samples, seed)

        # TODO: add some noise, calculate a target quantity on the noised data
        # TODO: return targets and no-noised samples
        # TODO: extend interface for number of wavelengths/intervals involved in the calculation of the target


def generate_2(seed):

    velo = np.diff(seed, 1)
    acc = np.diff(velo, 1)

    n = 20
    acc = acc * (0.1 * np.random.rand(n, acc.shape[0]) + 0.9)
    acc_offset = (seed[-1] - seed[0]) - (velo[0] * (seed.shape[0] - 1))
    acc_multiples = np.expand_dims(np.arange(seed.shape[0] - 2, 0, -1), axis=0)
    acc_norm = acc_offset / np.sum(acc * acc_multiples, axis=1)
    acc = acc * np.expand_dims(acc_norm, axis=1)

    #cov = np.eye(seed.shape[0] - 2)
    #sample = np.random.multivariate_normal(np.zeros((seed.shape[0]-2)), cov, size=n, check_valid='warn')
    #sample = np.array(sample).reshape((n, seed.shape[0]-2))

    #probabilistic_mask = 0.2 * np.abs(sample) + 0.8
    #acc = acc * probabilistic_mask

    velocity = np.tile(velo, (n, 1))
    for i in range(acc.shape[1]):
        velocity[:, i + 1] = velocity[:, i] + acc[:, i]

    trajectory = np.tile(seed, (n, 1))
    trajectory[:, 0] = trajectory[:, 0] + 0.06 * np.random.rand(trajectory.shape[0])
    for i in range(velocity.shape[1]):
        trajectory[:, i + 1] = trajectory[:, i] + velocity[:, i]

    mask = np.all(trajectory >= 0, axis=1)
    trajectory = np.compress(mask, trajectory, axis=0)

    print(f'Generated {trajectory.shape[0]} spectra')
    plot_spectra(np.arange(trajectory.shape[1]),
                 trajectory,
                 "Reflectance",
                 "Wavenumber",
                 "synthetic data",
                 "./synthetic data.png")

seeds = np.load("./seeds.npy")
seed = seeds[0, :]
generate_2(seed)









