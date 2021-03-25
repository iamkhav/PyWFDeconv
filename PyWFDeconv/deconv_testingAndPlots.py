import numpy as np
import time
from functools import partial
from . import (
    convar,
    firdif,
    early_stops,
    plot_code_excerpts
)
# import torch
# import ray
# import torch.multiprocessing as TorchMP


def deconv_testing(cal_data, ROI=None):
    """
    Using this for runtime analysis, no cross-validation (odd/even traces)

    :param cal_data:
    :param ROI:
    :return:
    """
    # start = time.time()

    # a more convenient (and faster) scaling to work with
    cal_data = cal_data * 100

    # calcium decay rate (single neuron, based on 40Hz measurments in Gcamp6f mice)
    gamma_40hz = 0.97

    # Benches
    # Openblas
    # test_traces = np.random.rand(800,200)             # 50s @ 52%
    # test_traces = np.random.rand(800,100)             # 30s @ 51%

    # test_traces = np.random.rand(400,800)               # 85s @ 40%
    # test_traces = np.random.rand(200,800)             # 26s @ 40%
    # test_traces = np.random.rand(100,800)             # 12s @ 36%           # Slicing and multiple convar execs seem to be the best way
    # test_traces = np.random.rand(50,800)                # 6.5s @ 22%
    # test_traces = np.random.rand(25,800)                # 4s @ 16%

    # test_traces = np.random.rand(100,3200)              # 75s @ 35%
    # test_traces = np.random.rand(4500,1)               # 1071s full convar        || 7s Mid convar
    # test_traces = np.random.rand(450,1)               #


    #Todo ever since correcting the gamma**exp loop the cuda_direct version is much quicker. Need new benches

    # Cuda
    # test_traces = np.random.rand(500,500)               # 59s
    # test_traces = np.random.rand(400,800)               # 62s
    # test_traces = np.random.rand(100,3200)               # 1.2s Mark || 1.5s Mid || 21.5s Convar
    # test_traces = np.random.rand(1024,1024)               # 1.3s Mark       || 9.3s Mid     || 320s Convar
    # test_traces = np.random.rand(200,3200)               # 1.3s Mark || 1.8s Mid || 65s Convar


    # Cuda direct
    # test_traces = np.random.rand(200,3200)               # 65s
    # test_traces = np.random.rand(4500,5)               # 402s full convar        || Mid convar 181s
    test_traces = np.random.rand(4500,1)               # New time: 1|7|28.6
    test_traces = np.random.rand(2000,1)               #
    # test_traces = np.random.rand(4500,2)               # 1.2s Mark || 7s Mid || 228s Convar
    # test_traces = np.random.rand(25,800)                # 1.2s Mark || 1.4s Mid || 4.75s Convar


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


    # search over a range of lambda/smoothing values to find the best one
    # all_lambda = [80, 40, 20, 10, 7, 5, 3, 2, 1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]      # Original Lambdas
    # all_lambda = [1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    all_lambda = [1]




    for k in range(0, len(all_lambda)):
        _lambda = all_lambda[k]

        # r, r1, beta0 = convar.convar_half_torch(test_traces, gamma, _lambda)

        # r, r1, beta0 = convar.convar_np_blas(test_traces, gamma, _lambda)
        # r, r1, beta0 = convar.convar_torch_cuda_direct(test_traces, gamma, _lambda)
        r, r1, beta0 = convar.convar_half_torch_cuda_direct(test_traces, gamma, _lambda)
        # r, r1, beta0 = convar.convar_torch_cuda(test_traces, gamma, _lambda)


    # end = time.time()
    # print(end - start)

def deconv_for_plots(cal_data, ROI=None):
    """
    Using this for Plots in Thesis
    :param cal_data:
    :param ROI:
    :return:
    """
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
        # r, r1, beta0 = plot_code_excerpts.meanGradient_over_t(odd_traces, gamma, _lambda)
        r, r1, beta0 = plot_code_excerpts.average_per_frame(odd_traces, gamma, _lambda)

