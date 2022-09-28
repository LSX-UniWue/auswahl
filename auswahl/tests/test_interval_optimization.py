import numpy as np
import pytest

from auswahl import optimize_intervals
from functools import partial


def test_exceptions():

    cases = [
        partial(optimize_intervals, 0, 1, np.ones((100,))),  # n_intervals 0
        partial(optimize_intervals, -1, 1, np.ones((100,))),  # n_intervals negative
        partial(optimize_intervals, 1, 0, np.ones((100,))),  # interval_width 0
        partial(optimize_intervals, 1, -1, np.ones((100,))),  # interval_width negative
        partial(optimize_intervals, 1.5, 1, np.ones((100,))),  # n_intervals non integer
        partial(optimize_intervals, 1, 1.5, np.ones((100,))),  # interval width non integer
        partial(optimize_intervals, 1, 1, np.ones((100, 2))),  # array of rank greater than one
        partial(optimize_intervals, 1, 100, np.ones((99,)))  # inconsistent array length
    ]

    for i, func in enumerate(cases):
        print(f'{i}')
        with pytest.raises(ValueError):
            func()


def test_optimization():
    scores = np.array([0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,0,0,0,0,0,0,0,0,5,5,5,5,5])
    gt = [5, 13, 26]
    _, interval_starts = optimize_intervals(3, 8, scores)
    assert (gt == interval_starts)
