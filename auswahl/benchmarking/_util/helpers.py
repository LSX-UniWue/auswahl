
import pickle

from typing import List


class Selection:

    """
        Helper class allowing the handling to lists of different lengths in a single pandas DataFrame.
    """

    def __init__(self, selected_features: List = None):
        self.selected_features = selected_features

    def __repr__(self):
        return str(self.selected_features)

    def __str__(self):
        return self.__repr__()


def load_pod(file_path: str):
    """
        TODO

    Parameters
    ----------
    file_path

    Returns
    -------

    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)
