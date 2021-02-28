import numpy as np
from torch.multiprocessing import Process, Pool
"""from ray.util.multiprocessing import Pool as RayPool
import ray"""
# import torch.multiprocessing as TorchMP
import convar
import time
# import torch
from functools import partial


def deconv_testing(cal_data, ROI=None):
    """
    Using this for runtime analysis, no cross-validation (odd/even traces)

    :param cal_data:
    :param ROI:
    :return:
    """
    start = time.time()

    # a more convenient (and faster) scaling to work with
    cal_data = cal_data * 100

    # calcium decay rate (single neuron, based on 40Hz measurments in Gcamp6f mice)
    gamma_40hz = 0.97

    # benches with openblas
    # test_traces = np.random.rand(800,200)             # 50s @ 52%
    # test_traces = np.random.rand(800,100)             # 30s @ 51%

    # test_traces = np.random.rand(200,800)             # 26s @ 40%
    test_traces = np.random.rand(100,800)             # 12s @ 36%           # Slicing and multiple convar execs seem to be the best way
    # test_traces = np.random.rand(100,3200)              # 75s @ 35%
    # test_traces = np.random.rand(400,800)               # 85s @ 40%


    # test_traces = np.random.rand(50,200)
    # Take cal_data from 0 or 1 to end, stepping in 2 -Amon
    # test_traces = cal_data[0::2]
    # test_traces = cal_data[0:50, 0:50]
    # test_traces = cal_data[:,0::2]

    print("Input shape:", np.shape(test_traces))
    print("Input total elements:", np.shape(test_traces)[0] * np.shape(test_traces)[1])

    # the calcium decay is needed to be fitted for 20hz of the even/odd traces
    ratio = 0.5
    gamma = 1 - (1 - gamma_40hz) / ratio

    # number of points in each odd/even calcium trace
    T = np.shape(test_traces)[0]
    rep = np.shape(test_traces)[1]

    # search over a range of lambda/smoothing values to find the best one
    # all_lambda = [80, 40, 20, 10, 7, 5, 3, 2, 1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]      # Original Lambdas
    # all_lambda = [1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    all_lambda = [1]

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

        r, r1, beta0 = convar.convar_np_openblas(test_traces, gamma, _lambda)


        # Leaving this part to see if I/O sizes differ

        # calculating the changes in spiking rate in each deconvolve trace
        r_diff = np.diff(r[1:], axis=0)
        # calculating the penalty in each trace
        penalty_size_convar[k] = np.mean(np.square(r_diff), axis=0)
        # reconstruct the calcium
        c_odd = np.matmul(Dinv, np.vstack((r1, r)))



    end = time.time()
    print(end - start)




def deconv(cal_data, ROI=None):
    start = time.time()

    # a more convenient (and faster) scaling to work with
    cal_data = cal_data * 100

    # calcium decay rate (single neuron, based on 40Hz measurments in Gcamp6f mice)
    gamma_40hz = 0.97

    # Take cal_data from 0 or 1 to end, stepping in 2 -Amon
    odd_traces = cal_data[0::2]
    even_traces = cal_data[1::2]

    # the calcium decay is needed to be fitted for 20hz of the even/odd traces
    ratio = 0.5
    gamma = 1 - (1 - gamma_40hz) / ratio

    # number of points in each odd/even calcium trace
    T = np.shape(odd_traces)[0]
    rep = np.shape(odd_traces)[1]

    # search over a range of lambda/smoothing values to find the best one
    # all_lambda = [80, 40, 20, 10, 7, 5, 3, 2, 1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]      # Original Lambdas
    # all_lambda = [1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    all_lambda = [1]

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

        r, r1, beta0 = convar.convar_np(odd_traces, gamma, _lambda)
        # calculating the changes in spiking rate in each deconvolve trace
        r_diff = np.diff(r[1:], axis=0)
        # calculating the penalty in each trace
        penalty_size_convar[k] = np.mean(np.square(r_diff), axis=0)
        # reconstruct the calcium
        c_odd = np.matmul(Dinv, np.vstack((r1, r)))
        calcium_dif_convar[k] = np.mean(np.abs(c_odd+beta0-even_traces), axis=0)

    temp = np.mean(calcium_dif_convar, axis=1)
    min_error_convar = np.min(temp)
    best_lambda_convar_indx = np.argmin(temp)
    best_lambda_convar = all_lambda[best_lambda_convar_indx]

    print("Min error Convar:", min_error_convar)
    print("Best Lambda:", best_lambda_convar)

    end = time.time()
    print(end - start)



def deconv_multicore(cal_data, ROI=None):
    start = time.time()

    # a more convenient (and faster) scaling to work with
    cal_data = cal_data * 100

    # calcium decay rate (single neuron, based on 40Hz measurments in Gcamp6f mice)
    gamma_40hz = 0.97

    # Take cal_data from 0 or 1 to end, stepping in 2 -Amon
    odd_traces = cal_data[0::2]
    even_traces = cal_data[1::2]

    # the calcium decay is needed to be fitted for 20hz of the even/odd traces
    ratio = 0.5
    gamma = 1 - (1 - gamma_40hz) / ratio

    # number of points in each odd/even calcium trace
    T = np.shape(odd_traces)[0]
    rep = np.shape(odd_traces)[1]

    # search over a range of lambda/smoothing values to find the best one
    all_lambda = [80, 40, 20, 10, 7, 5, 3, 2, 1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    all_lambda = [20, 10, 7, 5, 3, 2, 1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

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


    results = []
    # partial_f = partial(convar.convar_half_torch, odd_traces, gamma)
    partial_f = partial(convar.convar_np_openblas, odd_traces, gamma)
    with Pool(8) as p:
        results = p.map(partial_f, all_lambda)


    c = 0
    for k in results:
        # Unpacking variables from multiprocessing
        r, r1, beta0 = k[0], k[1], k[2]
        # calculating the changes in spiking rate in each deconvolve trace
        r_diff = np.diff(r[1:], axis=0)
        # calculating the penalty in each trace
        penalty_size_convar[c] = np.mean(np.square(r_diff), axis=0)
        # reconstruct the calcium
        c_odd = np.matmul(Dinv, np.vstack((r1, r)))
        calcium_dif_convar[c] = np.mean(np.abs(c_odd+beta0-even_traces), axis=0)
        c += 1

    temp = np.mean(calcium_dif_convar, axis=1)
    min_error_convar = np.min(temp)
    best_lambda_convar_indx = np.argmin(temp)
    best_lambda_convar = all_lambda[best_lambda_convar_indx]

    print("Min error Convar:", min_error_convar)
    print("Best Lambda:", best_lambda_convar)

    end = time.time()
    print(end - start)


