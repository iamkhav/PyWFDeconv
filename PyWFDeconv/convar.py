import numpy as np
from scipy import linalg
# import torch
import time
from math import ceil
import matplotlib.pyplot as plt
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


def convar_np(
    y, gamma, _lambda,
    init_out_matrix_method="firdif", init_output_mat=None,
    early_stop_bool=True, early_stop_metric_f=None, early_stop_threshold=None, return_stop_iter=False,
    num_iters=10000,
    adapt_lr_bool=False, gradient_rollback=False,
    printers=True
        ):
    """
        convar is a straight translation from matlab into numpy with some additional features.
        -Amon
    """
    # Set some default arguments
    if(early_stop_threshold==None):
        early_stop_threshold = 0.00003
        # early_stop_threshold = 0.00001            # Used this before normalizing input data in wfd.deconvolve
    if(early_stop_metric_f==None):
        early_stop_metric_f = early_stops.mean_abs

    if(not adapt_lr_bool and gradient_rollback):
        print("Adapt LR deactivated but Gradient Rollback activated!")
        print("Setting Gradient Rollback to False!!!")
        gradient_rollback = False
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
    s_start = s

    # Adaptive LR, increase right away -Amon
    if (adapt_lr_bool):
        # Start with a higher lr
        if (0.1 <= _lambda <= 1):
            s = s * helpers.scale_to(_lambda, 4, 1.5, 1, 0.1)
        elif (_lambda < 0.1):
            s = s * 1.5
        else:
            s = s * 4

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
    did_we_early_stop = False
    true_scaling_gradient = np.empty(1)
    # metric_gradient_i = np.empty(1)
    metric_gradient_i = 0
    # metric_gradient_prev_i = np.empty(1)
    metric_gradient_prev_i = 0
    metric_gradient_bigger_counter = 0
    rollback_counter = 0

    timer = 0
    # deconvolution
    for i in range(0, num_iters):
        # # # Numpy Matmuls
        # Ar = np.matmul(A, r)
        # tmAr = (tildey - Ar)
        # # t_0 = time.time()
        # At_tmAr = np.matmul(np.transpose(A), tmAr)
        # # timer += time.time() - t_0
        # Zr = np.matmul(Z, r)

        # Scipy BLAS Matmuls
        Ar = linalg.blas.sgemm(1, A, r)
        tmAr = (tildey - Ar)
        At_tmAr = linalg.blas.sgemm(1, np.transpose(A), tmAr)
        Zr = linalg.blas.sgemm(1, Z, r)


        if(adapt_lr_bool):
            # Using this because adapt_lr modifies s. That means the gradient is smaller/bigger because of the modified lr (s). We have to take an invariant s. -Amon
            true_scaling_gradient = s_start * At_tmAr - s_start * _lambda * Zr

            # Early Stop -Amon
            if (early_stop_bool and early_stop_metric_f(true_scaling_gradient) < early_stop_threshold):
                early_stopped_at = i
                did_we_early_stop = True

                # Conservative gradient descent at the end (to ensure that we're not leaving a local minimum) -Amon
                x = r + true_scaling_gradient
                r = x
                r[r < 0] = 0
                r[0] = x[0]
                break


        if (not gradient_rollback):
            # Calc Gradient -Amon
            gradient = s * At_tmAr - s * _lambda * Zr

            # Gradient Descent -Amon
            x = r + gradient
            r = x
            r[r < 0] = 0
            r[0] = x[0]

        if(adapt_lr_bool):
            # Gradient Comparison -Amon
            metric_gradient_i = early_stop_metric_f(true_scaling_gradient)
            if(i > 0):
                # if(metric_gradient_i > (metric_gradient_prev_i + 0.01 * metric_gradient_prev_i)):
                if(metric_gradient_i > (metric_gradient_prev_i)):
                    # Current gradient bigger than last one, BAD devolopment, decrease LR
                    # s = (s + s_start) / 2
                    s *= 0.95
                    # s *= 0.1

                    # Rollback r
                    # print("Rolling back..")
                    rollback_counter += 1
                    metric_gradient_bigger_counter += 1
                    # if(metric_gradient_bigger_counter >= 1):
                    #     print(metric_gradient_bigger_counter)
                    #     print("Current", metric_gradient_i)
                    #     print("Past", metric_gradient_prev_i)

                else:
                    # Current gradient smaller than, equal or within a range of last one, GOOD development, increase LR
                    # (Note: Equal or within a range could mean that we are close to convergence)
                    metric_gradient_bigger_counter = 0
                    if(gradient_rollback):
                        # Calc Gradient -Amon
                        gradient = s * At_tmAr - s * _lambda * Zr

                        # Gradient Descent -Amon
                        x = r + gradient
                        r = x
                        r[r < 0] = 0
                        r[0] = x[0]
                    s *= 1.01
                    # s *= 4


        # # NoAdaptLr Early Stop -Amon
        if((not adapt_lr_bool) and early_stop_bool and (early_stop_metric_f(gradient) < early_stop_threshold)):
            early_stopped_at = i
            did_we_early_stop = True
            break

        # Gradient Comparison -Amon
        if(adapt_lr_bool):
            metric_gradient_prev_i = metric_gradient_i

    r_final = r[1:]
    r1 = r[0:1]
    beta_0 = np.mean(y - np.matmul(Dinv, r), axis=0)

    # print(timer)

    if(printers):
        print("------------------------------------------------------")
        print("Convar stats")
        print(f"{'Mid convar time: ':^40} {round(mid - start, 2)}s")
        convar_time = time.time()-start
        print(f"{'Convar time:':^40} {round(convar_time, 2)}s")

        # To print the Metric Gradient, we have to fill the variable when it's not filled because of adapt_lr_bool=False
        if(not adapt_lr_bool):
            metric_gradient_i = early_stop_metric_f(gradient)
        print(f"{'Metric Gradient:':^40} {metric_gradient_i}")
        # print(_lambda)
        # print(f"Rollback Counter: {rollback_counter}")

        if(early_stop_bool and did_we_early_stop):
            print(f"{'Early stop at iteration:':^40} {early_stopped_at}")

            if(early_stopped_at>0):
                print(f"{'Estimated time w / o Early Stop:':^40} {round(convar_time * (1 / (early_stopped_at / num_iters)), 2)}s")

    if(return_stop_iter):
        return r_final,r1,beta_0, early_stopped_at
    else:
        return r_final,r1,beta_0



#
#
# def convar_half_torch(
#     y, gamma, _lambda,
#     init_out_matrix_method = "firdif", init_output_mat=None,
#     earlyStop_bool=True, earlyStop_f=early_stops.mean_threshold_torch,
#     num_iters=10000,
#     adapt_lr_bool=False, operation_order = 3
#         ):
#     """
#     Using numpy to initialize the matrices, then converting them into pytorch tensors.
#     Returning Numpy arrays at the end.
#     -Amon
#
#     :param y:
#     :param gamma:
#     :param _lambda:
#     :return:
#     """
#     start = time.time()
#
#     T = np.shape(y)[0]
#     P = np.identity(T) - 1 / T * np.ones((T,T))
#     tildey = np.matmul(P, y)
#
#     # will be used later to reconstruct the calcium from the deconvoled rates
#     Dinv = np.zeros((T, T))
#
#     for k in range(0, T):
#         for j in range(0, k + 1):
#             exp = (k - j)
#             Dinv[k][j] = gamma ** exp
#
#     A = np.matmul(P, Dinv)
#
#     L1 = np.zeros((T, T))
#     for i in range(0, T):
#         for j in range(0, T):
#             if(i >= 2 and j >= 1):
#                 if(i == j):
#                     L1[i][j] = 1
#                 if(i == j+1):
#                     L1[i][j] = -1
#
#     Z = np.matmul(np.transpose(L1), L1)
#
#     # large step size that ensures converges
#     s = 0.5 * ( (1-gamma)**2 / ( (1-gamma**T)**2 + (1-gamma)**2 * 4 * _lambda ) )
#
#     # initializing
#     # Initializing output matrix -Amon
#     if(init_out_matrix_method == "rand"):
#         r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
#     elif(init_out_matrix_method == "ones"):
#         r = np.ones((np.shape(y)[0], np.shape(y)[1]))
#     elif(init_out_matrix_method == "zeros"):
#         r = np.zeros((np.shape(y)[0], np.shape(y)[1]))
#     elif(init_out_matrix_method == "point5"):
#         r = np.zeros((np.shape(y)[0], np.shape(y)[1])) + 0.5
#     elif(init_out_matrix_method == "firdif"):
#         r_a, r_b, _ = firdif.firdif_np(y, gamma, 3)
#         r = np.concatenate((r_b, r_a))
#     elif(init_out_matrix_method == "input"):
#         r = init_output_mat
#     else:
#         raise Exception("init_out_matrix Argument not set correctly")
#
#     mid = time.time()
#     early_stopped_at = 1
#     did_we_early_stop = False
#
#     # Torch allocation
#     A = torch.from_numpy(A)
#     r = torch.from_numpy(r)
#     tildey = torch.from_numpy(tildey)
#     Z = torch.from_numpy(Z)
#     Dinv = torch.from_numpy(Dinv)
#     y = torch.from_numpy(y)
#
#     # deconvolution
#     for i in range(0, 10000):
#         Ar = torch.matmul(A, r)
#         tmAr = (tildey - Ar)
#         At_tmAr = torch.matmul(torch.transpose(A, 0, 1), tmAr)
#         Zr = torch.matmul(Z, r)
#         # Calc Gradient -Amon
#         gradient = s * At_tmAr - s * _lambda * Zr
#
#         # Gradient Descent -Amon
#         x = r + gradient
#         r = x
#         r[r < 0] = 0
#         r[0] = x[0]
#
#         # # Old Early Stop -Amon
#         # Don't use with adaptive LR
#         if(operation_order== 3 and earlyStop_bool and earlyStop_f(gradient)):
#             early_stopped_at = i
#             did_we_early_stop = True
#             break
#
#     r_final = r[1:]
#     r1 = r[0:1]
#     beta_0 = torch.mean(y - torch.matmul(Dinv, r), dim=0)
#
#     print("------------------------------------------------------")
#     print("Half Torch stats")
#     print(f"{'Mid convar time: ':^40} {round(mid - start, 2)}s")
#     convar_time = time.time()-start
#     print(f"{'Convar time:':^40} {round(convar_time, 2)}s")
#
#     if(earlyStop_bool and did_we_early_stop):
#         print(f"{'Early stop at iteration:':^40} {early_stopped_at}")
#
#         if(early_stopped_at>0):
#             print(f"{'Estimated time w / o Early Stop:':^40} {round(convar_time * (1 / (early_stopped_at / num_iters)), 2)}s")
#
#     return r_final.numpy(),r1.numpy(),beta_0.numpy()
#
#
#
# def convar_half_torch_cuda_direct(y, gamma, _lambda):
#     """
#         convar_torch implements torch data structures in order to use CUDA.
#         However before hand we use the CPU and numpy to initialize most stuff.
#         -Amon
#
#     :param y:
#     :param gamma:
#     :param _lambda:
#     :return:
#     """
#     # INIT
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if(device.type != "cuda"):
#         print("CUDA not available")
#         raise Exception("NO CUDA")
#
#     start = time.time()
#     # y = y.astype(np.float32)
#     print(y.dtype)
#     # y = torch.from_numpy(y).to(device)
#     y = torch.from_numpy(y).pin_memory().to(device, non_blocking=True)
#     print(y.dtype)
#
#     print("Mark time:", time.time() - start)
#
#
#     T = y.shape[0]
#     P = torch.eye(T, device=device) - 1 / T * torch.ones((T, T), device=device)
#     tildey = torch.matmul(P, y)
#
#     # will be used later to reconstruct the calcium from the deconvoled rates
#     # In the following part we have a huge bottleneck on GPU which is the ** operation. This is why we will transfer to GPU after initializing Dinv with double for loop
#     # Also numpy is much quicker than torch in this case
#     Dinv = np.zeros((T, T))
#
#     #Bottleneck Start
#     for k in range(0, T):
#         for j in range(0, k + 1):
#             exp = (k - j)
#             Dinv[k][j] = gamma ** exp
#     #Bottleneck End
#     Dinv = torch.from_numpy(Dinv).to(device)
#
#     A = torch.matmul(P, Dinv)
#
#     L1 = np.zeros((T, T))
#     for i in range(0, T):
#         for j in range(0, T):
#             if(i >= 2 and j >= 1):
#                 if(i == j):
#                     L1[i][j] = 1
#                 if(i == j+1):
#                     L1[i][j] = -1
#     L1transposed = torch.from_numpy(np.transpose(L1)).to(device)
#     L1 = torch.from_numpy(L1).to(device)
#
#     Z = torch.matmul(L1transposed, L1)
#
#     # large step size that ensures converges
#     s = 0.5 * ((1 - gamma) ** 2 / ((1 - gamma ** T) ** 2 + (1 - gamma) ** 2 * 4 * _lambda))
#
#     # deconvolution
#     # initializing
#     # r = np.random.rand(y.shape[0], y.shape[1])
#     r = torch.ones((y.shape[0], y.shape[1]), device=device)  # Test line for consistency instead of randomness
#
#     mid = time.time()
#     # All code until here is very light
#
#     for i in range(0, 10000):
#         Ar = torch.matmul(A, r)
#         tmAr = (tildey - Ar)
#         At_tmAr = torch.matmul(torch.transpose(A, 0, 1), tmAr)
#         Zr = torch.matmul(Z, r)
#         gradient = s * At_tmAr - s * _lambda * Zr
#         x = r + gradient
#         r = x
#         r[r < 0] = 0
#         r[0] = x[0]
#
#     r_final = r[1:]
#     r1 = r[0:1]
#     beta_0 = torch.mean(y - torch.matmul(Dinv, r), dim=0)
#
#
#     print("------------------------------------------------------")
#     print("Half Torch CUDA Direct stats")
#     print(f"{'Mid convar time: ':^40} {round(mid - start, 2)}s")
#     convar_time = time.time()-start
#     print(f"{'Convar time:':^40} {round(convar_time, 2)}s")
#
#     return r_final.cpu().numpy(),r1.cpu().numpy(),beta_0.cpu().numpy()
#
