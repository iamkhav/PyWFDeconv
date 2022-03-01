import time
import numpy as np
from . import helpers

"""
firdif from Merav Stern's Git.
Used in initializing the output matrix in convar instead of randomizing.

Original Docstring:
% This function implements the first diffrences method
% Inputs:  y - row vector, the measured fluorescence trace; 
% if y is a matrix each row in the matrix is treated as a fluorescence trace
% y - size Txn
% gamma - number, the calcium decay between two measurment points
% smt - number, smoothing (number of points to use) on the algorith rate result    
% Returns the deconvolved rate r_final (T-1xn)
% the first point of the calcium r1 (1xn)
% and the offset beta 0 (1Xn)
"""

def firdif_best_window(data, gamma):
    # start = time.time()

    # a more convenient (and faster) scaling to work with
    # data = data * 100

    # calcium decay rate (single neuron, based on 40Hz measurments in Gcamp6f mice)
    # gamma_40hz = 0.97

    # Take cal_data from 0 or 1 to end, stepping in 2 -Amon
    odd_traces = data[0::2]
    even_traces = data[1::2]

    # the calcium decay is needed to be fitted for 20hz of the even/odd traces
    ratio = 0.5
    gamma = 1 - (1 - gamma) / ratio

    # number of points in each odd/even calcium trace
    T = np.shape(odd_traces)[0]
    rep = np.shape(odd_traces)[1]

    # search over a range of lambda/smoothing values to find the best one
    all_lambda = [3,5,7,9,11,13,15,17]

    # will be used later to reconstruct the calcium from the deconvoled rates
    Dinv = np.zeros((T, T))

    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp

    # saving the results
    # here the penalty (l2) is the same as the fluctuations (l2)
    penalty_size_convar = np.zeros((len(all_lambda), rep))
    calcium_dif_convar = np.zeros((len(all_lambda), rep))

    for k in range(0, len(all_lambda)):
        _lambda = all_lambda[k]

        r, r1, beta0 = firdif_np(odd_traces, gamma, smt=_lambda, need_beta0_bool=True)


        # calculating the changes in spiking rate in each deconvolve trace
        # r_diff = np.diff(r[1:], axis=0)
        # calculating the penalty in each trace
        # penalty_size_convar[k] = np.mean(np.square(r_diff), axis=0)
        # reconstruct the calcium
        c_odd = np.matmul(Dinv, np.vstack((r1, r)))
        calcium_dif_convar[k] = np.mean(np.abs(c_odd + beta0 - even_traces), axis=0)

    temp = np.mean(calcium_dif_convar, axis=1)
    # min_error_convar = np.min(temp)
    best_lambda_convar_indx = np.argmin(temp)
    best_lambda_convar = all_lambda[best_lambda_convar_indx]

    # print("------------------------------------------------------")
    # print("------------------------------------------------------")
    # print("Min error Convar:", min_error_convar)
    # print("Best Lambda:", best_lambda_convar)
    # print(temp)
    # end = time.time()
    # print("Full Deconv time:", end - start)
    return best_lambda_convar

def firdif_np(y, gamma, smt=3, shift_to_allPos_bool=True, need_beta0_bool=False, printers=False):
    """


    :param y:
    :param gamma:
    :param smt: Smoothing factor, should be an odd number to replicate exact matlab procedure
    :param shift_to_allPos_bool: Shifts all values in r_final and r1 to positive values (default:True)
    :param need_beta0_bool: Returns beta0 as 3rd return value if set to True, else 0 (default:False)
    :return: returns r_final, r1 and beta0. Beta0 is however skipped if need_beta0_bool is False
    """
    start = time.time()
    T = np.shape(y)[0]
    D = np.identity(T)

    for i in range(0, T):
        for j in range(0, T-1):
            if(i == j):
                D[i+1][j] = -gamma

    # deconvolve
    r = np.matmul(D, y)

    # smoothing the results (without r[0] which is c[0] and not a spiking rate)
    a = r[2 : int(2 + np.floor(smt/2)), :]
    a = np.flipud(a)

    b = r[int(np.shape(r)[0] - np.floor(smt/2) - 1) : -1, :]
    b = np.flipud(b)

    r_long = np.concatenate((a, r[1:], b))

    r_smoothed = np.zeros_like(r_long)
    for i in range(0, np.shape(y)[1]):
        r_smoothed[:, i] = helpers.moving_average(r_long[:, i], smt)

    ind_start = int(np.floor(smt/2))
    ind_end = np.shape(r_smoothed)[0] - int(np.floor(smt/2))
    r2toT = r_smoothed[ind_start :  ind_end, :]

    # shifting r to be non negative (for r(2)... r(T))
    # while keeping the whole rate trace (for the inference beta)
    r_shifted = np.concatenate((r[0:1, :], r2toT))

    if(shift_to_allPos_bool):
        for i in range(0, np.shape(y)[1]):
            if(np.min(r2toT[:, i]) < 0):
                r_shifted[:, i] = r_shifted[:, i] - np.min(r2toT[:, i])

    r_final = r_shifted[1:]
    r1 = r_shifted[0:1]

    if(not need_beta0_bool):
        # To save some time if beta0 is not needed
        if(printers):
            print("------------------------------------------------------")
            print("Firdif stats")
            print("Time:", time.time() - start)
        return r_final, r1, 0

    Dinv = np.zeros((T, T))

    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp

    beta0 = np.mean(y - np.matmul(Dinv, r_shifted), axis=0)

    if(printers):
        print("------------------------------------------------------")
        print("Firdif stats")
        print("Time:", time.time() - start)
    return r_final, r1, beta0