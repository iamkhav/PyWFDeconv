import numpy as np

"""
Trying hard to use Cython or Pythran for performance...
"""

# pythran export moving_average(float[], int)
def moving_average(inny, span):
    """
    Should be an exact copy of the matlab smooth function when used with "moving" setting (which is default).
    :return: The smoother array
    """
    outy = []
    outy.append(inny[0])
    i = 0
    while(len(inny)+1 > i + span):
        outy.append(sum(inny[i:i+span]) / span)
        i += 1

    outy.append(inny[-1])
    return np.array(outy)