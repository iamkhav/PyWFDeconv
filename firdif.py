import time
import numpy as np
"""
firdif from Merav Stern's Git.
Used in initializing the output matrix in convar instead of randomizing.
Incomplete
"""


def firdif_np(y, gamma, _lambda):
    start = time.time()
    T = np.shape(y)[0]
    D = np.identity(T)

    for i in range(0, T):
        for j in range(0, T-1):
            if(i == j):
                D[i+1][j] = -gamma

    r = np.matmul(D, y)
    print(r)

    return