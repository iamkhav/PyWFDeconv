import numpy as np
from scipy import linalg
import torch
import time
from . import (
    firdif,
    early_stops,
    helpers
)
# import warnings
# import cupy as cp
# import gc
"""
Advanced Versions of Convar with all features.
-Amon
"""


def convar_np(y, gamma, _lambda, init_out_matrix_method = "firdif", init_output_mat=None, earlyStop_bool=True, earlyStop_f=early_stops.mean_threshold, num_iters = 10000):
    """
    convar is a straight translation from matlab into numpy with some additional features.
    -Amon
    """
    start = time.time()
    T = np.shape(y)[0]
    P = np.identity(T) - 1 / T * np.ones((T,T))
    tildey = np.matmul(P, y)

    # will be used later to reconstruct the calcium from the deconvoled rates
    Dinv = np.zeros((T, T))

    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp

    A = np.matmul(P, Dinv)

    L1 = np.zeros((T, T))
    for i in range(0, T):
        for j in range(0, T):
            if(i >= 2 and j >= 1):
                if(i == j):
                    L1[i][j] = 1
                if(i == j+1):
                    L1[i][j] = -1

    Z = np.matmul(np.transpose(L1), L1)

    # large step size that ensures converges
    s = 0.5 * ( (1-gamma)**2 / ( (1-gamma**T)**2 + (1-gamma)**2 * 4 * _lambda ) )
    # s = s*2

    # deconvolution
    # Initializing output matrix -Amon
    if(init_out_matrix_method == "rand"):
        r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    elif(init_out_matrix_method == "ones"):
        r = np.ones((np.shape(y)[0], np.shape(y)[1]))
    elif(init_out_matrix_method == "zeros"):
        r = np.zeros((np.shape(y)[0], np.shape(y)[1]))
    elif(init_out_matrix_method == "point5"):
        r = np.zeros((np.shape(y)[0], np.shape(y)[1])) + 0.5
    elif(init_out_matrix_method == "firdif"):
        r_a, r_b, _ = firdif.firdif_np(y, gamma, 3)
        r = np.concatenate((r_b, r_a))
    elif(init_out_matrix_method == "input"):
        r = init_output_mat
    else:
        raise Exception("init_out_matrix Argument not set correctly")

    mid = time.time()
    early_stopped_at = 1

    for i in range(0, num_iters):
        Ar = np.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = np.matmul(np.transpose(A), tmAr)
        Zr = np.matmul(Z, r)
        gradient = s * At_tmAr - s * _lambda * Zr
        x = r + gradient
        r = x
        r[r < 0] = 0
        r[0] = x[0]

        # Early Stop -Amon
        if(earlyStop_bool and earlyStop_f(gradient)):
            print(f"Early Stop at {i} iterations")
            early_stopped_at = i
            # print("Abs Mean Grad", np.abs(np.mean(gradient)))
            # print("Max, Min", np.max(gradient), np.min(gradient))
            # print("Std Grad ddof0", np.std(gradient, ddof=0))
            # print("Std Grad ddof1", np.std(gradient, ddof=1))
            break

        # Adaptive LR -Amon
        # Biggest Gradient
        # if(i==0):
        #     big_grad = np.abs(np.mean(gradient))
        # s = s * helpers.scale_to(np.abs(np.mean(gradient)), 2, 0.5)
        # print(helpers.scale_to(np.abs(np.mean(gradient)), 2, 0.5))
        # s = s * helpers.scale_to(np.abs(np.mean(gradient)), 1, 0.5, max_x=big_grad)
        s = s * 0.99


    r_final = r[1:]
    r1 = r[0:1]
    beta_0 = np.mean(y - np.matmul(Dinv, r), axis=0)

    print("------------------------------------------------------")
    print("Numpy stats")
    print(f"{'Mid convar time: ':^40} {round(mid - start, 2)}s")
    convar_time = time.time()-start
    print(f"{'Convar time:':^40} {round(convar_time, 2)}s")
    if(earlyStop_bool):
        print(f"{'Estimated time w / o Early Stop:':^40} {round(convar_time * (1 / (early_stopped_at / num_iters)), 2)}s")

    return r_final,r1,beta_0


def convar_half_torch_cuda_direct(y, gamma, _lambda):
    """
    convar_torch implements torch data structures in order to use CUDA.
    However before hand we use the CPU and numpy to initialize most stuff.
    -Amon

    :param y:
    :param gamma:
    :param _lambda:
    :return:
    """
    # INIT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(device.type != "cuda"):
        print("CUDA not available")
        raise Exception("NO CUDA")

    start = time.time()
    # y = y.astype(np.float32)
    print(y.dtype)
    # y = torch.from_numpy(y).to(device)
    y = torch.from_numpy(y).pin_memory().to(device, non_blocking=True)
    print(y.dtype)

    print("Mark time:", time.time() - start)


    T = y.shape[0]
    P = torch.eye(T, device=device) - 1 / T * torch.ones((T, T), device=device)
    tildey = torch.matmul(P, y)

    # will be used later to reconstruct the calcium from the deconvoled rates
    # In the following part we have a huge bottleneck on GPU which is the ** operation. This is why we will transfer to GPU after initializing Dinv with double for loop
    # Also numpy is much quicker than torch in this case
    Dinv = np.zeros((T, T))

    #Bottleneck Start
    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp
    #Bottleneck End
    Dinv = torch.from_numpy(Dinv).to(device)

    A = torch.matmul(P, Dinv)

    L1 = np.zeros((T, T))
    for i in range(0, T):
        for j in range(0, T):
            if(i >= 2 and j >= 1):
                if(i == j):
                    L1[i][j] = 1
                if(i == j+1):
                    L1[i][j] = -1
    L1transposed = torch.from_numpy(np.transpose(L1)).to(device)
    L1 = torch.from_numpy(L1).to(device)

    Z = torch.matmul(L1transposed, L1)

    # large step size that ensures converges
    s = 0.5 * ((1 - gamma) ** 2 / ((1 - gamma ** T) ** 2 + (1 - gamma) ** 2 * 4 * _lambda))

    # deconvolution
    # initializing
    # r = np.random.rand(y.shape[0], y.shape[1])
    r = torch.ones((y.shape[0], y.shape[1]), device=device)  # Test line for consistency instead of randomness

    mid = time.time()
    # All code until here is very light

    for i in range(0, 10000):
        Ar = torch.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = torch.matmul(torch.transpose(A, 0, 1), tmAr)
        Zr = torch.matmul(Z, r)
        gradient = s * At_tmAr - s * _lambda * Zr
        x = r + gradient
        r = x
        r[r < 0] = 0
        r[0] = x[0]

    r_final = r[1:]
    r1 = r[0:1]
    beta_0 = torch.mean(y - torch.matmul(Dinv, r), dim=0)


    print("------------------------------------------------------")
    print("Half Torch CUDA Direct stats")
    print(f"{'Mid convar time: ':^40} {round(mid - start, 2)}s")
    convar_time = time.time()-start
    print(f"{'Convar time:':^40} {round(convar_time, 2)}s")

    return r_final.cpu().numpy(),r1.cpu().numpy(),beta_0.cpu().numpy()

