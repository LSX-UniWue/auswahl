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

        n = 250 #  TODO: the generation process is probabilistic as of now, fix that with n_samples

        # TODO: proof of concept
        acc = acc * (0.2 * np.random.rand(n, acc.shape[0]) + 0.8)

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
        for i in range(velocity.shape[1]):
            trajectory[:, i + 1] = trajectory[:, i] + velocity[:, i]

        diffs = np.abs(trajectory[:, -1] - seed[-1]) <= 0.05
        trajectory = np.compress(diffs, trajectory, axis=0)

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

    n = 200
    acc = acc * (0.2 * np.random.rand(n, acc.shape[0]) + 0.8)
    #cov = np.eye(seed.shape[0] - 2)
    #sample = np.random.multivariate_normal(np.zeros((seed.shape[0]-2)), cov, size=n, check_valid='warn')
    #sample = np.array(sample).reshape((n, seed.shape[0]-2))

    #probabilistic_mask = 0.2 * np.abs(sample) + 0.8
    #acc = acc * probabilistic_mask

    velocity = np.tile(velo, (n, 1))
    for i in range(acc.shape[1]):
        velocity[:, i + 1] = velocity[:, i] + acc[:, i]

    trajectory = np.tile(seed, (n,1))
    for i in range(velocity.shape[1]):
        trajectory[:, i + 1] = trajectory[:, i] + velocity[:, i]

    diffs = np.abs(trajectory[:, -1] - seed[-1]) <= 0.05
    trajectory = np.compress(diffs, trajectory, axis=0)

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









