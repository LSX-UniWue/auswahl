
import pickle

from ._data_handling import BenchmarkPOD


def load_pod(file_path: str):
    with open(file_path, 'rb') as file:
        return pickle.load(file)