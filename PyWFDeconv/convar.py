import numpy as np
from scipy import linalg
import torch
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
    init_out_matrix_method = "firdif", init_output_mat=None,
    early_stop_bool=True, early_stop_f=early_stops.mean_abs, early_stop_threshold=0.00001,return_stop_iter=False,
    num_iters=10000,
    adapt_lr_bool=False
        ):
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
    s_start = s

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
    metric_gradient_i = np.empty(1)
    metric_gradient_prev_i = np.empty(1)
    metric_gradient_bigger_counter = 0

    # Adaptive LR, start increase -Amon
    if (adapt_lr_bool):
        # Start with a higher lr
        if (0.1 <= _lambda <= 1):
            s = s * helpers.scale_to(_lambda, 4, 1.5, 1, 0.1)
            # s = s * helpers.scale_to(_lambda, 4, 1.5, 1, 0.1)
        elif (_lambda < 0.1):
            s = s * 1.5
        else:
            s = s * 4

    # deconvolution
    for i in range(0, num_iters):
        # # Numpy Matmuls
        Ar = np.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = np.matmul(np.transpose(A), tmAr)
        Zr = np.matmul(Z, r)

        # # Scipy BLAS Matmuls
        # Ar = linalg.blas.sgemm(1, A, r)
        # tmAr = (tildey - Ar)
        # At_tmAr = linalg.blas.sgemm(1, np.transpose(A), tmAr)
        # Zr = linalg.blas.sgemm(1, Z, r)



        if(adapt_lr_bool):
            # Using this because adapt_lr modifies s. This way the gradient is smaller because of the smaller lr (s) and we have to take an invariant s. -Amon

            true_scaling_gradient = s_start * At_tmAr - s_start * _lambda * Zr

            # Early Stop -Amon
            if (early_stop_bool and early_stop_f(true_scaling_gradient) < early_stop_threshold):
                early_stopped_at = i
                did_we_early_stop = True

                # Conservative gradient descent -Amon
                x = r + true_scaling_gradient
                r = x
                r[r < 0] = 0
                r[0] = x[0]
                break

        # Calc Gradient -Amon
        gradient = s * At_tmAr - s * _lambda * Zr

        # Gradient Descent -Amon
        x = r + gradient
        r = x
        r[r < 0] = 0
        r[0] = x[0]

        # Gradient Comparison -Amon
        if(adapt_lr_bool):
            metric_gradient_i = early_stop_f(gradient)
            if(i > 0):
                # if(metric_gradient_i > (metric_gradient_prev_i + 0.01 * metric_gradient_prev_i)):
                if(metric_gradient_i > (metric_gradient_prev_i)):
                    metric_gradient_bigger_counter += 1
                    if(metric_gradient_bigger_counter >= 5):
                        print(metric_gradient_bigger_counter)
                        print("Current", metric_gradient_i)
                        print("Past", metric_gradient_prev_i)
                else:
                    metric_gradient_bigger_counter = 0



        # # NoAdaptLr Early Stop -Amon
        if((not adapt_lr_bool) and early_stop_bool and (early_stop_f(gradient) < early_stop_threshold)):
            early_stopped_at = i
            did_we_early_stop = True
            break

        # Gradient Comparison -Amon
        if(adapt_lr_bool):
            metric_gradient_prev_i = metric_gradient_i

    r_final = r[1:]
    r1 = r[0:1]
    beta_0 = np.mean(y - np.matmul(Dinv, r), axis=0)

    print("------------------------------------------------------")
    print("Numpy stats")
    print(f"{'Mid convar time: ':^40} {round(mid - start, 2)}s")
    convar_time = time.time()-start
    print(f"{'Convar time:':^40} {round(convar_time, 2)}s")

    if(early_stop_bool and did_we_early_stop):
        print(f"{'Early stop at iteration:':^40} {early_stopped_at}")

        if(early_stopped_at>0):
            print(f"{'Estimated time w / o Early Stop:':^40} {round(convar_time * (1 / (early_stopped_at / num_iters)), 2)}s")
        # print("Mean Abs Grad", np.mean(np.abs(true_scaling_gradient)))
        # print("Max, Min", np.max(true_scaling_gradient), np.min(true_scaling_gradient))
        # print("Std Grad ddof0", np.std(true_scaling_gradient, ddof=0))
        # print("Std Grad ddof1", np.std(true_scaling_gradient, ddof=1))

    if(return_stop_iter):
        return r_final,r1,beta_0, early_stopped_at
    else:
        return r_final,r1,beta_0



def old_convar_np(
    y, gamma, _lambda,
    init_out_matrix_method = "firdif", init_output_mat=None,
    early_stop_bool=True, early_stop_f=early_stops.mean_abs, early_stop_threshold=0.00001,return_stop_iter=False,
    num_iters=10000,
    adapt_lr_bool=False
        ):
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
    s_start = s

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
    metric_gradient_i = np.empty(1)
    metric_gradient_prev_i = np.empty(1)
    metric_gradient_bigger_counter = 0

    # deconvolution
    for i in range(0, num_iters):
        # print(i)
        # # Numpy Matmuls
        Ar = np.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = np.matmul(np.transpose(A), tmAr)
        Zr = np.matmul(Z, r)

        # # Scipy BLAS Matmuls
        # Ar = linalg.blas.sgemm(1, A, r)
        # tmAr = (tildey - Ar)
        # At_tmAr = linalg.blas.sgemm(1, np.transpose(A), tmAr)
        # Zr = linalg.blas.sgemm(1, Z, r)



        if(adapt_lr_bool):
            # Using this because adapt_lr modifies s. This way the gradient is smaller because of the smaller lr (s) and we have to take an invariant s. -Amon

            true_scaling_gradient = s_start * At_tmAr - s_start * _lambda * Zr

            # Early Stop -Amon
            if (early_stop_bool and early_stop_f(true_scaling_gradient) < early_stop_threshold):
                early_stopped_at = i
                did_we_early_stop = True

                # Conservative gradient descent -Amon
                x = r + true_scaling_gradient
                r = x
                r[r < 0] = 0
                r[0] = x[0]
                break

        # Calc Gradient -Amon
        gradient = s * At_tmAr - s * _lambda * Zr

        # Gradient Descent -Amon
        x = r + gradient
        r = x
        r[r < 0] = 0
        r[0] = x[0]

        # Gradient Comparison -Amon
        if(not adapt_lr_bool):
            metric_gradient_i = early_stop_f(gradient)
            if(i > 0):
                # if(metric_gradient_i > (metric_gradient_prev_i + 0.01 * metric_gradient_prev_i)):
                if(metric_gradient_i > (metric_gradient_prev_i)):
                    metric_gradient_bigger_counter += 1
                    if(metric_gradient_bigger_counter >= 5):
                        print(metric_gradient_bigger_counter)
                        print("Current", metric_gradient_i)
                        print("Past", metric_gradient_prev_i)
                else:
                    metric_gradient_bigger_counter = 0

        # Adaptive LR -Amon
        if(adapt_lr_bool):
            if(i % 1000 == 0):   # Start with a higher lr and increase every x iters (good values for x seem to be >500)
                if(0.1<=_lambda<=1):
                    s = s * helpers.scale_to(_lambda, 4, 1.5, 1, 0.1)
                    # s = s * helpers.scale_to(_lambda, 4, 1.5, 1, 0.1)
                elif(_lambda < 0.1):
                    s = s * 1.5
                else:
                    s = s * 4

        # # Old Early Stop -Amon
        if((not adapt_lr_bool) and early_stop_bool and (early_stop_f(gradient) < early_stop_threshold)):
            early_stopped_at = i
            did_we_early_stop = True
            break

        # Gradient Comparison -Amon
        if(not adapt_lr_bool):
            metric_gradient_prev_i = metric_gradient_i

    r_final = r[1:]
    r1 = r[0:1]
    beta_0 = np.mean(y - np.matmul(Dinv, r), axis=0)

    print("------------------------------------------------------")
    print("Numpy stats")
    print(f"{'Mid convar time: ':^40} {round(mid - start, 2)}s")
    convar_time = time.time()-start
    print(f"{'Convar time:':^40} {round(convar_time, 2)}s")

    if(early_stop_bool and did_we_early_stop):
        print(f"{'Early stop at iteration:':^40} {early_stopped_at}")

        if(early_stopped_at>0):
            print(f"{'Estimated time w / o Early Stop:':^40} {round(convar_time * (1 / (early_stopped_at / num_iters)), 2)}s")
        # print("Abs Mean Grad", np.abs(np.mean(true_scaling_gradient)))
        # print("Max, Min", np.max(true_scaling_gradient), np.min(true_scaling_gradient))
        # print("Std Grad ddof0", np.std(true_scaling_gradient, ddof=0))
        # print("Std Grad ddof1", np.std(true_scaling_gradient, ddof=1))

    if(return_stop_iter):
        return r_final,r1,beta_0, early_stopped_at
    else:
        return r_final,r1,beta_0




def convar_cow(y, gamma, _lambda):
    """
        This convar function first does an experimental run with adaptive lr and then a conservative run on the first runs output with conventional settings.
        Very experimental.
        Worst case is both runs using all iterations instead of early stopping (which would probably yield more accurate results tho).
        Best case however is the first run early stopping somewhere and the second run stopping at iteration 0 (because the calculated gradient is so small).

    :param y:
    :param gamma:
    :param _lambda:
    :return:
    """
    fr, fr1, _ = convar_np(y, gamma, _lambda)
    stitched_fr = np.concatenate((fr1, fr))
    # mr, mr1, mbeta0 = convar_np(y, gamma, _lambda, init_out_matrix_method="input", init_output_mat=stitched_fr)


    # r, r1, _, stop_iter = convar_np(y, gamma, _lambda, adapt_lr_bool=True)
    # # r, r1, _ = convar_np(y, gamma, _lambda, adapt_lr_bool=True, earlyStop_bool=False)
    #
    # stitched_r = np.concatenate((r1, r))
    # nr, nr1, nbeta0 = convar_np(y, gamma, _lambda, init_out_matrix_method="input", init_output_mat=stitched_r)


    #
    #
    r, r1, _, stop_iter = convar_np(y, gamma, _lambda, adapt_lr_bool=True, return_stop_iter=True)
    # r, r1, _ = convar_np(y, gamma, _lambda, adapt_lr_bool=True, earlyStop_bool=False)
    stop_iter = ceil(stop_iter/10)
    stitched_r = np.concatenate((r1, r))
    # r, r1, _ = convar_np(y, gamma, _lambda, init_out_matrix_method="input", init_output_mat=stitched_r, num_iters=stop_iter, early_stop_bool=False)
    # stitched_r = np.concatenate((r1, r))
    nr, nr1, nbeta0 = convar_np(y, gamma, _lambda, init_out_matrix_method="input", init_output_mat=stitched_r)



    # PLOT
    start_Frame = 10

    average_original = []
    for i in y[start_Frame:]:
        # average_original.append(np.mean(i) / np.std(y))
        average_original.append(np.mean(i))
    average_original = helpers.normalize_1_0(average_original)


    average_firdif = []
    for i in fr[start_Frame:]:
        # average_firdif.append(np.mean(i) / np.std(r_firdif))
        average_firdif.append(np.mean(i))
    average_firdif = helpers.normalize_1_0(average_firdif)

    average_firdifhalfLR = []
    for i in r[start_Frame:]:
        # average_firdifLR.append(np.mean(i) / np.std(r_firdifLR))
        average_firdifhalfLR.append(np.mean(i))
    average_firdifhalfLR = helpers.normalize_1_0(average_firdifhalfLR)

    average_firdifLR = []
    for i in nr[start_Frame:]:
        # average_firdifLR.append(np.mean(i) / np.std(r_firdifLR))
        average_firdifLR.append(np.mean(i))
    average_firdifLR = helpers.normalize_1_0(average_firdifLR)



    # Plot 1
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["figure.figsize"] = (8, 6)

    plt.plot(average_original, label="Original", linewidth=2, alpha=0.5, color="b")
    plt.plot(average_firdif, label="Standard FirDif Convar", linewidth=2, color="orange")
    # plt.plot(average_firdifhalfLR, label="Half FirDif Adaptive LR", linewidth=2, color="plum")
    plt.plot(average_firdifLR, label="Double Convar + Adaptive LR", linewidth=2, color="magenta")

    plt.ylabel("Mean of Output")
    plt.xlabel("Frames")
    plt.title("Convar: Mean of Output per Frame")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    diffo = np.array(average_firdif) - np.array(average_firdifLR)
    print(f"Mean Difference: {np.mean(diffo)}")

    # Plot 2 Difference - Useless?
    # diffo = np.array(average_firdif) - np.array(average_firdifLR)
    # plt.plot(diffo, label="a = Standard, b = DoubleConvar+AdaptLR", linewidth=2, color="magenta")
    #
    # plt.ylabel("Mean of Output (a) - Mean of Output (b)")
    # plt.xlabel("Frames")
    # plt.title("Convar: Difference of Mean of Output per Frame")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.close()


    return nr, nr1, nbeta0


def convar_half_torch(
    y, gamma, _lambda,
    init_out_matrix_method = "firdif", init_output_mat=None,
    earlyStop_bool=True, earlyStop_f=early_stops.mean_threshold_torch,
    num_iters=10000,
    adapt_lr_bool=False, operation_order = 3
        ):
    """
    Using numpy to initialize the matrices, then converting them into pytorch tensors.
    Returning Numpy arrays at the end.
    -Amon

    :param y:
    :param gamma:
    :param _lambda:
    :return:
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

    # initializing
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

    # Torch allocation
    A = torch.from_numpy(A)
    r = torch.from_numpy(r)
    tildey = torch.from_numpy(tildey)
    Z = torch.from_numpy(Z)
    Dinv = torch.from_numpy(Dinv)
    y = torch.from_numpy(y)

    # deconvolution
    for i in range(0, 10000):
        Ar = torch.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = torch.matmul(torch.transpose(A, 0, 1), tmAr)
        Zr = torch.matmul(Z, r)
        # Calc Gradient -Amon
        gradient = s * At_tmAr - s * _lambda * Zr

        # Gradient Descent -Amon
        x = r + gradient
        r = x
        r[r < 0] = 0
        r[0] = x[0]

        # # Old Early Stop -Amon
        # Don't use with adaptive LR
        if(operation_order== 3 and earlyStop_bool and earlyStop_f(gradient)):
            early_stopped_at = i
            did_we_early_stop = True
            break

    r_final = r[1:]
    r1 = r[0:1]
    beta_0 = torch.mean(y - torch.matmul(Dinv, r), dim=0)

    print("------------------------------------------------------")
    print("Half Torch stats")
    print(f"{'Mid convar time: ':^40} {round(mid - start, 2)}s")
    convar_time = time.time()-start
    print(f"{'Convar time:':^40} {round(convar_time, 2)}s")

    if(earlyStop_bool and did_we_early_stop):
        print(f"{'Early stop at iteration:':^40} {early_stopped_at}")

        if(early_stopped_at>0):
            print(f"{'Estimated time w / o Early Stop:':^40} {round(convar_time * (1 / (early_stopped_at / num_iters)), 2)}s")

    return r_final.numpy(),r1.numpy(),beta_0.numpy()



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

