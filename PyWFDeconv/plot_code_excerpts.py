import time
import numpy as np
import matplotlib.pyplot as plt
from . import (
    early_stops,
    firdif,
    helpers
)


def meanGradient_over_t(y, gamma, _lambda, init_out_matrix = "rand", earlyStop_bool=False, earlyStop_f=early_stops.mean_threshold, num_iters=10000):
    """

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

    # deconvolution
    # Initializing output matrix -Amon
    r = np.random.rand(np.shape(y)[0], np.shape(y)[1])


    mid = time.time()
    rand_plot = []
    for i in range(0, num_iters):
        Ar = np.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = np.matmul(np.transpose(A), tmAr)
        Zr = np.matmul(Z, r)
        gradient = s * At_tmAr - s * _lambda * Zr           # For Early Stop
        x = r + gradient
        r = x
        r[r < 0] = 0
        r[0] = x[0]
        rand_plot.append(np.abs(np.mean(gradient)))
        #Early Stop -Amon
        if(earlyStop_bool and earlyStop_f(gradient)):
            print(f"Early Stop at {i} iterations")
            break

    r = np.ones((np.shape(y)[0], np.shape(y)[1]))
    ones_plot = []
    for i in range(0, num_iters):
        Ar = np.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = np.matmul(np.transpose(A), tmAr)
        Zr = np.matmul(Z, r)
        gradient = s * At_tmAr - s * _lambda * Zr           # For Early Stop
        x = r + gradient
        r = x
        r[r < 0] = 0
        r[0] = x[0]
        ones_plot.append(np.abs(np.mean(gradient)))
        #Early Stop -Amon
        if(earlyStop_bool and earlyStop_f(gradient)):
            print(f"Early Stop at {i} iterations")
            break


    # r = np.zeros((np.shape(y)[0], np.shape(y)[1])) + 0.5
    # zp5_plot = []
    # for i in range(0, 10000):
    #     Ar = np.matmul(A, r)
    #     tmAr = (tildey - Ar)
    #     At_tmAr = np.matmul(np.transpose(A), tmAr)
    #     Zr = np.matmul(Z, r)
    #     gradient = s * At_tmAr - s * _lambda * Zr           # For Early Stop
    #     x = r + gradient
    #     r = x
    #     r[r < 0] = 0
    #     r[0] = x[0]
    #     zp5_plot.append(np.abs(np.mean(gradient)))
    #     #Early Stop -Amon
    #     if(earlyStop_bool and earlyStop_f(gradient)):
    #         print(f"Early Stop at {i} iterations")
    #         break



    r_a, r_b, _ = firdif.firdif_np(y, gamma, 3)
    r = np.concatenate((r_b, r_a))
    firdif_plot = []
    for i in range(0, num_iters):
        Ar = np.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = np.matmul(np.transpose(A), tmAr)
        Zr = np.matmul(Z, r)
        gradient = s * At_tmAr - s * _lambda * Zr           # For Early Stop
        x = r + gradient
        r = x
        r[r < 0] = 0
        r[0] = x[0]
        firdif_plot.append(np.abs(np.mean(gradient)))
        # #Early Stop -Amon
        # if(earlyStop_bool and earlyStop_f(gradient)):
        #     print(f"Early Stop at {i} iterations")
        #     break

        # Adaptive LR -Amon
        s = s * helpers.scale_to(np.abs(np.mean(gradient)), 2, 0.5)



    # Plot
    plt.axhline(y=0.00001, color="r", label="Early Stop Threshold", linestyle=":")
    plt.plot(rand_plot, label="Rand", alpha=0.8)
    plt.plot(ones_plot, label="Ones", alpha=0.8)
    plt.plot(firdif_plot, label="First-Difference", linewidth=2)
    plt.ylabel("Mean gradient")
    plt.xlabel("Iterations")
    plt.yscale("log")
    plt.title("Gradient Progression with varying Output Inits")
    plt.legend()
    plt.show()



    print("------------------------------------------------------")
    print("Numpy stats")
    print("Mid convar time:", mid-start)
    print("Convar time:", time.time()-start)


    r_final = r[1:]
    r1 = r[0:1]
    beta_0 = np.mean(y - np.matmul(Dinv, r), axis=0)

    return r_final,r1,beta_0



def average_per_frame(y, gamma, _lambda, init_out_matrix = "rand", earlyStop_bool=True, earlyStop_f=early_stops.mean_threshold, num_iters=10000):
    """

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
    print("LR:", s)
    # # deconvolution
    # # Initializing output matrix -Amon
    r_rand = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    rand_plot = []
    for i in range(0, num_iters):
        Ar = np.matmul(A, r_rand)
        tmAr = (tildey - Ar)
        At_tmAr = np.matmul(np.transpose(A), tmAr)
        Zr = np.matmul(Z, r_rand)
        gradient = s * At_tmAr - s * _lambda * Zr           # For Early Stop
        x = r_rand + gradient
        r_rand = x
        r_rand[r_rand < 0] = 0
        r_rand[0] = x[0]
        rand_plot.append(np.abs(np.mean(gradient)))
        #Early Stop -Amon
        # if(earlyStop_bool and earlyStop_f(gradient)):
        #     print(f"Early Stop at {i} iterations")
        #     break

    r_ones = np.ones((np.shape(y)[0], np.shape(y)[1]))
    ones_plot = []
    for i in range(0, num_iters):
        Ar = np.matmul(A, r_ones)
        tmAr = (tildey - Ar)
        At_tmAr = np.matmul(np.transpose(A), tmAr)
        Zr = np.matmul(Z, r_ones)
        gradient = s * At_tmAr - s * _lambda * Zr           # For Early Stop
        x = r_ones + gradient
        r_ones = x
        r_ones[r_ones < 0] = 0
        r_ones[0] = x[0]
        ones_plot.append(np.abs(np.mean(gradient)))
        # #Early Stop -Amon
        # if(earlyStop_bool and earlyStop_f(gradient)):
        #     print(f"Early Stop at {i} iterations")
        #     break


    # r = np.zeros((np.shape(y)[0], np.shape(y)[1])) + 0.5
    # zp5_plot = []
    # for i in range(0, 10000):
    #     Ar = np.matmul(A, r)
    #     tmAr = (tildey - Ar)
    #     At_tmAr = np.matmul(np.transpose(A), tmAr)
    #     Zr = np.matmul(Z, r)
    #     gradient = s * At_tmAr - s * _lambda * Zr           # For Early Stop
    #     x = r + gradient
    #     r = x
    #     r[r < 0] = 0
    #     r[0] = x[0]
    #     zp5_plot.append(np.abs(np.mean(gradient)))
    #     #Early Stop -Amon
    #     if(earlyStop_bool and earlyStop_f(gradient)):
    #         print(f"Early Stop at {i} iterations")
    #         break



    r_a, r_b, _ = firdif.firdif_np(y, gamma, 3)
    r_firdif = np.concatenate((r_b, r_a))
    firdif_plot = []
    for i in range(0, num_iters):
        Ar = np.matmul(A, r_firdif)
        tmAr = (tildey - Ar)
        At_tmAr = np.matmul(np.transpose(A), tmAr)
        Zr = np.matmul(Z, r_firdif)
        gradient = s * At_tmAr - s * _lambda * Zr           # For Early Stop
        x = r_firdif + gradient
        r_firdif = x
        r_firdif[r_firdif < 0] = 0
        r_firdif[0] = x[0]
        firdif_plot.append(np.abs(np.mean(gradient)))
        # #Early Stop -Amon
        # if(earlyStop_bool and earlyStop_f(gradient)):
        #     print(f"Early Stop at {i} iterations")
        #     break

        # Adaptive LR -Amon
        # s = s * helpers.scale_to(np.abs(np.mean(gradient)), 2, 0.5)
        # s = s * 0.99999


    r_a, r_b, _ = firdif.firdif_np(y, gamma, 3)
    r_firdifLR = np.concatenate((r_b, r_a))
    firdif_plot = []
    for i in range(0, num_iters):
        Ar = np.matmul(A, r_firdifLR)
        tmAr = (tildey - Ar)
        At_tmAr = np.matmul(np.transpose(A), tmAr)
        Zr = np.matmul(Z, r_firdifLR)
        gradient = s * At_tmAr - s * _lambda * Zr           # For Early Stop
        x = r_firdifLR + gradient
        r_firdifLR = x
        r_firdifLR[r_firdifLR < 0] = 0
        r_firdifLR[0] = x[0]
        firdif_plot.append(np.abs(np.mean(gradient)))
        # #Early Stop -Amon
        # if(earlyStop_bool and earlyStop_f(gradient)):
        #     print(f"Early Stop at {i} iterations")
        #     break

        # Adaptive LR -Amon
        s = s * helpers.scale_to(np.abs(np.mean(gradient)), 2, 0.5)
        s = s * 0.9


    # Smooth r
    # for i in range(0, np.shape(r)[0]):
    #     r[i] = helpers.moving_average(r[i], 3)


    # Organizing Data for Plots
    start_Frame = 10

    average_original = []
    for i in y[start_Frame:]:
        # average_original.append(np.mean(i) / np.std(y))
        average_original.append(np.mean(i))
    average_original = helpers.normalize_1_minus1(average_original)

    average_ones = []
    for i in r_ones[start_Frame:]:
        # average_firdif.append(np.mean(i) / np.std(r_ones))
        average_ones.append(np.mean(i))
    average_ones = helpers.normalize_1_minus1(average_ones)

    average_rand = []
    for i in r_rand[start_Frame:]:
        # average_firdif.append(np.mean(i) / np.std(average_rand))
        average_rand.append(np.mean(i))
    average_rand = helpers.normalize_1_minus1(average_rand)

    average_firdif = []
    for i in r_firdif[start_Frame:]:
        # average_firdif.append(np.mean(i) / np.std(r_firdif))
        average_firdif.append(np.mean(i))
    average_firdif = helpers.normalize_1_minus1(average_firdif)

    average_firdifLR = []
    for i in r_firdifLR[start_Frame:]:
        # average_firdifLR.append(np.mean(i) / np.std(r_firdifLR))
        average_firdifLR.append(np.mean(i))
    average_firdifLR = helpers.normalize_1_minus1(average_firdifLR)


    # Plot
    plt.rcParams.update({'font.size': 16})
    plt.plot(average_original, label="Original", linewidth=2, alpha=0.5)
    plt.plot(average_firdif, label="FirDif", linewidth=2)
    plt.plot(average_firdifLR, label="FirDif Adaptive LR", linewidth=2)
    plt.plot(average_rand, label="Random", linewidth=2)
    plt.plot(average_ones, label="Ones", linewidth=2)
    plt.ylabel("Mean/Std Output")
    plt.xlabel("Frames")
    plt.title("Convar Outputs")
    plt.legend()
    plt.show()




    # One Pixel over Frames
    pixel_index = 6
    plt.plot(y[10:, pixel_index] / np.std(y), label="Original", linewidth=2)
    plt.plot(r_firdif[10:, pixel_index] / np.std(r_firdif), label="FirDif", linewidth=2)
    plt.legend()
    plt.show()


    print("------------------------------------------------------")
    print("Numpy stats")
    print("Convar time:", time.time()-start)


    # r_final = r[1:]
    # r1 = r[0:1]
    # beta_0 = np.mean(y - np.matmul(Dinv, r), axis=0)

    return 0,0,0

