import numpy as np
import time
import torch
from . import convar_deprecated
# from functools import partial
# from torch.multiprocessing import Process, Pool
"""from ray.util.multiprocessing import Pool as RayPool
import ray"""
# import torch.multiprocessing as TorchMP



def deconv_torch_jit(cal_data):
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
    # all_lambda = [1, 2]

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

        start = time.time()
        r, r1, beta0 = convar_deprecated.convar_torch_jit(torch.from_numpy(odd_traces), torch.tensor(gamma), torch.tensor(_lambda))
        r, r1, beta0 = r.numpy(), r1.numpy(), beta0.numpy()
        end = time.time()
        print("Time:", end-start)

        # calculating the changes in spiking rate in each deconvolve trace
        r_diff = np.diff(r[1:], axis=0)
        # calculating the penalty in each trace
        penalty_size_convar[k] = np.mean(np.square(r_diff), axis=0)
        # reconstruct the calcium
        c_odd = np.matmul(Dinv, np.vstack((r1, r)))
        calcium_dif_convar[k] = np.mean(np.abs(c_odd+beta0-even_traces), axis=0)
        # Done token

    temp = np.mean(calcium_dif_convar, axis=1)
    min_error_convar = np.min(temp)
    best_lambda_convar_indx = np.argmin(temp)
    best_lambda_convar = all_lambda[best_lambda_convar_indx]

    print("Min error Convar:", min_error_convar)
    print(best_lambda_convar_indx)
    print("Best Lambda:", best_lambda_convar)


    # partial_f = partial(convar.convar_torch_jit, odd_traces, gamma)
    # with Pool(6) as p:
    #     p.map(partial_f, all_lambda)

    end = time.time()
    print(end - start)





"""
def deconv_multicore_ray(cal_data):
    ray.init()
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
    # all_lambda = [1, 2]

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

    # Todo checken ob ray auf windows überhaupt geht
    # for k in range(0, len(all_lambda)):
    #     _lambda = all_lambda[k]
    #     r, r1, beta0 = convar.convar_half_torch(odd_traces, gamma, _lambda)
    #
    #     # calculating the changes in spiking rate in each deconvolve trace
    #     r_diff = np.diff(r[1:], axis=0)
    #     # calculating the penalty in each trace
    #     penalty_size_convar[k] = np.mean(np.square(r_diff), axis=0)
    #     # reconstruct the calcium
    #     c_odd = np.matmul(Dinv, np.vstack((r1, r)))
    #     calcium_dif_convar[k] = np.mean(np.abs(c_odd+beta0-even_traces), axis=0)
    #
    # temp = np.mean(calcium_dif_convar, axis=1)
    # min_error_convar = np.min(temp)
    # best_lambda_convar_indx = np.argmin(temp)
    # best_lambda_convar = all_lambda[best_lambda_convar_indx]
    #
    # print("Min error Convar:", min_error_convar)
    # print(best_lambda_convar_indx)
    # print("Best Lambda:", best_lambda_convar)


    partial_f = partial(convar.convar_half_torch, odd_traces, gamma)
    # p = RayPool(2)
    # # p.map(partial_f, all_lambda)
    # for result in p.map(f, range(100)):
    #     print(result)

    end = time.time()
    print(end - start)"""