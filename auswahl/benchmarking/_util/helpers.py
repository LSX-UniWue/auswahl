
import pickle

from typing import List


class Selection:

    """Helper class allowing the handling of lists of different lengths in a single pandas DataFrame.

       Parameters
       ----------
       selected_features: List[int], default=None
            List of feature indices wrapped by this instance

       Attributes
       ----------
       features: List[int]
            List of feature indices contained in the instance
    """

    def __init__(self, selected_features: List[int] = None):
        self.features = selected_features

    def __repr__(self):
        return str(self.features)

    def __str__(self):
        return self.__repr__()


def load_data_handler(file_path: str):
    """
        Function to load a pickled instance of :class:`~auswahl.benchmarking.DataHandler`

    Parameters
    ----------
    file_path: str
        path to the file

    Returns
    -------
    loaded data: DataHandler
        Returns the loaded instance of type :class:`~auswahl.benchmarking.DataHandler`

    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)
