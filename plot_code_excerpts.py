import time
import numpy as np
import early_stops
import firdif
import matplotlib.pyplot as plt


def convar_np(y, gamma, _lambda, init_out_matrix = "rand", earlyStop_bool=False, earlyStop_f=early_stops.mean_threshold):
    """
    convar is a straight translation from matlab into numpy with some additional features.
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

    # deconvolution
    # Initializing output matrix -Amon
    r = np.random.rand(np.shape(y)[0], np.shape(y)[1])


    mid = time.time()
    rand_plot = []
    for i in range(0, 10000):
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
    for i in range(0, 10000):
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
    for i in range(0, 10000):
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
        #Early Stop -Amon
        if(earlyStop_bool and earlyStop_f(gradient)):
            print(f"Early Stop at {i} iterations")
            break
    # elif (init_out_matrix == "zeros"):
    #     r = np.zeros((np.shape(y)[0], np.shape(y)[1]))
    # elif (init_out_matrix == "point5"):
    #     r = np.zeros((np.shape(y)[0], np.shape(y)[1])) + 0.5
    # elif (init_out_matrix == "firdif"):
    #     r_a, r_b, _ = firdif.firdif_np(y, gamma, 3)
    #     r = np.concatenate((r_b, r_a))

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