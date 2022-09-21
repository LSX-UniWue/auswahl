import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def optimize_intervals(n_intervals: int, interval_width: int, feature_scores: np.array):
    """The algorithm calculates the optimal non-overlapping placement of n_intervals of width interval_width into the
    range of features. The feature scores are specified in feature_scores (greater better). The algorithm can be used
    for instance to turn every point selection algorithm yielding a score for each feature
    (such as :class:`~auswahl.VIP`) into an interval selection algorithm. The runtime of the algorithm is :math:`O(kn)`, with
    k being n_intervals and n feature_scores.size

    Parameters
    ----------
    n_intervals: int
        Number of intervals to be placed.

    interval_width: int
        Width of the intervals.

    feature_scores: np.array of shape (n, )
        Array of scores of variables (greater better).

    Returns
    -------
    tuple: (float, List[int])
        Tuple of overall score of the interval placement, list of interval starts.
    """

    if not isinstance(n_intervals, int):
        raise ValueError(f'optimize_intervals requires an integer for argument n_intervals. Got {type(n_intervals)}')
    if not isinstance(interval_width, int):
        raise ValueError(f'optimize_intervals requires an integer for argument interval_width.'
                         f'Got {type(interval_width)}')
    if n_intervals <= 0:
        raise ValueError(f'optimize_intervals requires a positive integer for argument n_intervals. Got {n_intervals}')
    if interval_width <= 0:
        raise ValueError(f'optimize_intervals requires a positive integer for argument interval_width.'
                         f'Got {interval_width}')
    if len(feature_scores.shape) != 1:
        raise ValueError(f'optimize_intervals requires an array of rank one. Got rank {len(feature_scores.shape)}')
    if feature_scores.size < n_intervals * interval_width:
        raise ValueError(f'optimize_intervals requires an array of feature scores with at least n_interval *'
                         f'interval_width. Required at least {n_intervals * interval_width}, got {feature_scores.size}')

    # costs[i]: total score of the interval starting at i
    intervals = sliding_window_view(feature_scores, interval_width)
    interval_scores = np.sum(intervals, axis=1)

    # table[i, k] specifies the score of allocating in the range of features {0, ..., i} k intervals
    # initialize with sentinels
    # introduce index -1 sentinel into the first axis of the table
    table = -1000000 * np.ones((feature_scores.size + 1, n_intervals + 1))
    table[:, 0] = 0  # zero costs for no intervals
    interval_starts = [[[] for _ in range(n_intervals + 1)] for _ in range(feature_scores.size)]

    for i in range(interval_width - 1, feature_scores.size):
        for k in range(1, min((i + 1) // interval_width, n_intervals) + 1):
            score_incl_i = interval_scores[i - interval_width + 1] + table[i - interval_width, k - 1]
            if score_incl_i > table[i-1, k]:
                table[i, k] = score_incl_i
                interval_starts[i][k] = interval_starts[i - interval_width][k - 1] + [i - interval_width + 1]
            else:
                table[i, k] = table[i - 1, k]
                interval_starts[i][k] = interval_starts[i - 1][k].copy()

    return table[feature_scores.size - 1, n_intervals], interval_starts[feature_scores.size - 1][n_intervals]
