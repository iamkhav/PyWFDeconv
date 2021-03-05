import numpy as np
from scipy import linalg
import torch
import time
# import gc
from numba import jit
import helpers



@jit(nopython=True)
def convar_numba(y, gamma, _lambda):
    """
    Don't use this function.
    It doesn't return correct values due to having certain numpy functions not supported by numba.
    -Amon

    :param y:
    :param gamma:
    :param _lambda:
    :return:
    """
    T = np.shape(y)[0]
    P = np.identity(T) - 1 / T * np.ones((T,T))
    tildey = P @ y

    # will be used later to reconstruct the calcium from the deconvoled rates
    Dinv = np.zeros((T, T))

    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp

    A = P @ Dinv

    L1 = np.zeros((T, T))
    for i in range(0, T):
        for j in range(0, T):
            if(i >= 2 and j >= 1):
                if(i == j):
                    L1[i][j] = 1
                if(i == j+1):
                    L1[i][j] = -1

    Z = np.transpose(L1) @ L1

    # large step size that ensures converges
    s = 0.5 * ( (1-gamma)**2 / ( (1-gamma**T)**2 + (1-gamma)**2 * 4 * _lambda ) )

    # deconvolution
    # initializing
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = np.ones((np.shape(y)[0], np.shape(y)[1]))  # Test line for consistency instead of randomness


    for i in range(0, 10000):
        Ar = A @ r
        tmAr = (tildey - Ar)
        At_tmAr = np.transpose(A) @ tmAr
        Zr = Z @ r
        x = r + s*At_tmAr - s*_lambda*Zr
        r = x
        # r[r < 0] = 0
        r[0] = x[0]
    r_final = r[1:]
    r1 = r[0]
    beta_0 = np.mean(y - Dinv @ r)

    print("------------------------------------------------------")
    print("Numpy stats")

    return r_final,r1,beta_0