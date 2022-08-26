import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def optimize_intervals(n_intervals: int, interval_width: int, feature_scores: np.array):
    """ The algorithm calculates the optimal non-overlapping placement of n_intervals of width interval_width into the
    range of features. The feature scores are specified in feature_scores (greater better). The algorithm can be used
    for instance to turn every point selection algorithm yielding a score for each feature (such as VIP) into an
    interval selection algorithm

    Parameters
    ----------
    n_intervals: int
        Number of intervals to be placed

    interval_width: int
        Width of the intervals

    feature_scores: np.array of shape (n, )
        Array of scores of variables (greater better)

    Returns
    -------
    tuple: (float, List[int])
        tuple of overall score of the interval placement, list of interval starts
    """

    if not isinstance(n_intervals, int):
        raise ValueError(f'optimize_intervals requires an integer for argument n_intervals. Got {type(n_intervals)}')
    if not isinstance(interval_width, int):
        raise ValueError(f'optimize_intervals requires an integer for argument interval_width. Got {type(interval_width)}')
    if n_intervals <= 0:
        raise ValueError(f'optimize_intervals requires a positive integer for argument n_intervals. Got {n_intervals}')
    if interval_width <= 0:
        raise ValueError(f'optimize_intervals requires a positive integer for argument interval_width. Got {interval_width}')
    if len(feature_scores.shape) != 1:
        raise ValueError(f'optimize_intervals requires an array of rank one. Got rank {len(feature_scores.shape)}')
    if feature_scores.size < n_intervals * interval_width:
        raise ValueError(f'optimize_intervals requires an array of feature scores with at least n_interval * interval_width.'
                         f'Required at least {n_intervals * interval_width}, got {feature_scores.size}')

    # costs[i]: total score of the interval starting at i
    cumsum = np.concatenate([np.array([0]), np.cumsum(feature_scores)])
    intervals = sliding_window_view(cumsum, interval_width + 1)
    interval_scores = intervals[:, -1] - intervals[:, 0]

    # table[i, k] specifies the score of allocating in the range {0, ..., i} k intervals
    # initalize with sentinels
    table = -1000000 * np.ones((feature_scores.size, n_intervals + 1))
    table[:, 0] = 0  # zero costs for no intervals
    interval_starts = [[[] for _ in range(n_intervals + 1)] for _ in range(feature_scores.size)]

    for i in range(interval_width - 1, feature_scores.size):
        for k in range(1, min((i + 1) // interval_width, n_intervals) + 1):
            # iterate through placements of the last interval in the considered interval
            scores = [(p, interval_scores[p] + table[p - 1, k - 1]) for p in range(i - interval_width + 1,
                                                                                   (k - 1) * interval_width - 1, -1)]
            pos, score = max(scores, key=lambda x: x[1])
            table[i, k] = score
            interval_starts[i][k] = interval_starts[pos - 1][k - 1] + [pos]

    return table[feature_scores.size - 1, n_intervals], interval_starts[feature_scores.size - 1][n_intervals]
