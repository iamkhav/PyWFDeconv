import numpy as np

"""
Defaults:
mean_threshold : 0.00001
biggest_pog_neg_sum_threshold : 0.003
"""


def mean_threshold(gradient, threshold=0.00001):
    if(np.abs(np.mean(gradient)) < threshold):
        return True


def biggest_pos_neg_sum_threshold(gradient, threshold=0.0003):
    """
    Find biggest positive and biggest |negative|.
    If no negative values, 0 is returned.

    Using this as a metric for Early Stop.

    :param in_array: Some numpy array.
    :return: Biggest Pos, Biggest |Neg|
    """
    curr_bigpos = 0
    curr_bigneg = 0
    for x in np.ravel(gradient):
        # if(x > 0 and x > curr_bigpos):
        #     curr_bigpos = x
        # if(x < 0 and x < curr_bigneg):
        #     curr_bigneg = x
        if(x > 0):
            if(x > curr_bigpos):
                curr_bigpos = x
        else:
            if(x < curr_bigneg):
                curr_bigneg = x

    if(curr_bigpos + np.abs(curr_bigneg) < threshold):
        return True