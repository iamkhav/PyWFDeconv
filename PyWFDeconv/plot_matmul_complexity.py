import os
# Deactivate multithreading (doesn't really work without threadpoolctl)
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
from threadpoolctl import threadpool_limits
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from . import convar as convar
from scipy import linalg


"""
This file is completely aimed at testing runtime and real life appliance of theoretical complexity of the convar.
DON'T USE TO GET RESULTS!!!
"""


def convar_slim(y, gamma, _lambda, num_iters=10000):
    """
    Slim version with only 1 core usage for benchmarking.
    -Amon

    :param y:
    :param gamma:
    :param _lambda:
    :return:
    """
    start = time.time()
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

    Z = np.matmul(np.transpose(L1), L1)

    # large step size that ensures converges
    s = 0.5 * ( (1-gamma)**2 / ( (1-gamma**T)**2 + (1-gamma)**2 * 4 * _lambda ) )

    # deconvolution
    # initializing
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = np.ones((np.shape(y)[0], np.shape(y)[1]))

    mid = time.time()
    # Deactivate multithreading for all BLAS implementations
    with threadpool_limits(limits=1, user_api='blas'):
        for i in range(0, num_iters):
            Ar = linalg.blas.sgemm(1, A, r)
            tmAr = (tildey - Ar)
            At_tmAr = linalg.blas.sgemm(1, np.transpose(A), tmAr)
            Zr = linalg.blas.sgemm(1, Z, r)
            x = r + s*At_tmAr - s*_lambda*Zr
            r = x
            r[r < 0] = 0
            r[0] = x[0]
    # r_final = r[1:]
    # r1 = r[0:1]
    # beta_0 = np.mean(y - (Dinv @ r), axis=0)

    print("------------------------------------------------------")
    print("Numpy Slim stats")
    print(f"{'Mid convar time: ':^40} {round(mid - start, 2)}s")
    convar_time = time.time() - start
    print(f"{'Convar time:':^40} {round(convar_time, 2)}s")

    # return r_final,r1,beta_0
    return 0,0,0

def matrimuli(a, b):
    """Multiplication: Matrix times Matrix.
        m is the matrix with which to multiply.
        Return the result as a new Matrix4.
        Make sure that you do not change self or the other matrix.
        return this * m"""
    m = np.shape(a)[0]
    n = np.shape(a)[1]
    p = np.shape(b)[1]

    outMatrix = np.zeros((m,p))
    for x in range(0, m):
        for i in range(0, p):
            for j in range(0, n):
                outMatrix[x,i] += a[x,j] * b[j,i]
                # print(a[x,j] * b[j,i])
    return outMatrix


def convar_own_matmul(y, gamma, _lambda, num_iters=10000):
    """
    This version uses a selfmade matmul for testing purposes.
    -Amon

    :param y:
    :param gamma:
    :param _lambda:
    :return:
    """
    start = time.time()
    T = np.shape(y)[0]
    P = np.identity(T) - 1 / T * np.ones((T,T))
    tildey = matrimuli(P, y)

    # will be used later to reconstruct the calcium from the deconvoled rates
    Dinv = np.zeros((T, T))

    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp

    A = matrimuli(P,Dinv)

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
    # initializing
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = np.ones((np.shape(y)[0], np.shape(y)[1]))

    mid = time.time()
    # Deactivate multithreading for all BLAS implementations
    with threadpool_limits(limits=1, user_api='blas'):
        for i in range(0, num_iters):
            Ar = matrimuli(A, r)
            tmAr = (tildey - Ar)
            At_tmAr = matrimuli(np.transpose(A),tmAr)
            Zr = matrimuli(Z, r)
            x = r + s*At_tmAr - s*_lambda*Zr
            r = x
            r[r < 0] = 0
            r[0] = x[0]
    # r_final = r[1:]
    # r1 = r[0:1]
    # beta_0 = np.mean(y - (Dinv @ r), axis=0)

    print("------------------------------------------------------")
    print("Numpy Slim stats")
    print(f"{'Mid convar time: ':^40} {round(mid - start, 2)}s")
    convar_time = time.time() - start
    print(f"{'Convar time:':^40} {round(convar_time, 2)}s")

    # return r_final,r1,beta_0
    return 0,0,0


def check_matmul_runtime(T_size, P_size, iters = 100):
    # Generate test matrices
    matrices_a = np.random.rand(iters, T_size, T_size)
    matrices_b = np.random.rand(iters, T_size, P_size)

    start = time.time()
    with threadpool_limits(limits=1, user_api='blas'):
        for a, b in zip(matrices_a, matrices_b):
            # np.matmul(a,b)
            Ar = linalg.blas.sgemm(1, a, b)

    total = time.time() - start
    print("------------------------------------------------------")
    print(T_size)
    print(f"{'Total time:':^40} {round(total, 2)}")
    print(f"{'Time per iter:':^40} {round(total/iters, 2)}")


def check_matmul_runtime_quicker(T_size, P_size, iters = 100):
    # Generate test matrices
    a = np.random.rand(T_size, T_size)
    b = np.random.rand(T_size, P_size)

    start = time.time()
    # with threadpool_limits(limits=1, user_api='blas'):
    #     for x in range(0, iters):
    #         # np.matmul(a,b)
    #         Ar = linalg.blas.sgemm(1, a, b)
    for x in range(0, iters):
        # np.matmul(a,b)
        Ar = linalg.blas.sgemm(1, a, b)
    total = time.time() - start
    print("------------------------------------------------------")
    print(T_size)
    print(f"{'Total time:':^40} {round(total, 2)}")
    print(f"{'Time per iter:':^40} {round(total/iters, 2)}")


def plot_or_bench_matmul_runtime():
    # Benchmark np.matmul
    numero_itero = 1000
    p_size = 400
    # np.matmul
    # check_matmul_runtime(25, p_size, iters=numero_itero)    # 0.06
    # check_matmul_runtime(50, p_size, iters=numero_itero)    # 0.06
    # check_matmul_runtime(100, p_size, iters=numero_itero)   # 0.18
    # check_matmul_runtime(200, p_size, iters=numero_itero)   # 0.62
    # check_matmul_runtime(400, p_size, iters=numero_itero)   # 2.47
    # check_matmul_runtime(800, p_size, iters=numero_itero)   # 9.21
    # check_matmul_runtime(1600, p_size, iters=numero_itero)  # 36.69


    # check_matmul_runtime_quicker(25, p_size, iters=numero_itero)          # 0.06
    # check_matmul_runtime_quicker(50, p_size, iters=numero_itero)          # 0.06
    # check_matmul_runtime_quicker(100, p_size, iters=numero_itero)         # 0.17
    # check_matmul_runtime_quicker(200, p_size, iters=numero_itero)         # 0.59
    # check_matmul_runtime_quicker(400, p_size, iters=numero_itero)         # 2.43
    # check_matmul_runtime_quicker(800, p_size, iters=numero_itero)         # 9.09
    # check_matmul_runtime_quicker(1600, p_size, iters=numero_itero)        # 35.58
    # check_matmul_runtime_quicker(3200, p_size, iters=numero_itero)        # 142.91
    # check_matmul_runtime_quicker(6400, p_size, iters=numero_itero/10)     #
    # check_matmul_runtime_quicker(12800, p_size, iters=numero_itero/100)   #

    # # Plot
    # plt.rcParams.update({'font.size': 13})
    # plt.rcParams["figure.figsize"] = (8, 6)
    # fig, ax = plt.subplots()
    #
    # benches = [0.06, 0.06, 0.17, 0.59, 2.43, 9.09, 35.58, 142.91]
    # x_ticks = ["25", "50", "100", "200", "400", "800", "1600", "3200"]
    # ranger = [25,50,100,200,400, 800, 1600, 3200]
    # plt.xticks(ranger,x_ticks, rotation=30)
    #
    # # ax.set_yscale("log", base=2)
    # # ax.yaxis.set_major_locator(ticker.MultipleLocator(22))
    #
    # # plt.plot(ranger, benches, color="r")
    # ax.plot(ranger, benches, color="r")
    # # plt.semilogy(ranger,benches, base=2)
    # plt.title(f"np.matmul runtime: Static P=400, Iters=1000, scaling T")
    # plt.ylabel("Time [s]")
    # plt.xlabel("T")
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    # #scipy blas
    # # 1000 iters in the end, check excel for benches Präsis/03.05/benches.xlsx
    # check_matmul_runtime_quicker(25, p_size, iters=5000)          #
    # check_matmul_runtime_quicker(50, p_size, iters=5000)          #
    # check_matmul_runtime_quicker(100, p_size, iters=5000)         #
    # check_matmul_runtime_quicker(200, p_size, iters=5000)         #
    # check_matmul_runtime_quicker(400, p_size, iters=500)         #
    # check_matmul_runtime_quicker(800, p_size, iters=500)         #
    # check_matmul_runtime_quicker(1600, p_size, iters=100)        #
    # check_matmul_runtime_quicker(3200, p_size, iters=100)        #
    # check_matmul_runtime_quicker(6400, p_size, iters=5)     #
    # check_matmul_runtime_quicker(12800, p_size, iters=5)   #

    # Plot WITHOUT multi
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["figure.figsize"] = (8, 6)
    fig, ax = plt.subplots()

    benches = [0.028, 0.046, 0.124, 0.374, 1.34, 5.74, 24.5, 108.8, 554, 2494]
    x_ticks = ["25", "50", "100", "200", "400", "800", "1600", "3200", "6400", "12800"]
    ranger = [25,50,100,200,400, 800, 1600, 3200, 6400, 12800]
    plt.xticks(ranger,x_ticks, rotation=90)

    # ax.set_yscale("log", base=2)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(22))

    # plt.plot(ranger, benches, color="r")
    ax.plot(ranger, benches, color="r")
    # plt.semilogy(ranger,benches, base=2)
    plt.title(f"scipy.matmul no multi runtime: Static P=400, Iters=1000, scaling T")
    plt.ylabel("Time [s]")
    plt.xlabel("T")
    plt.tight_layout()
    plt.show()
    plt.close()

    #
    # ------------------------------
    # Plot WITH multi
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["figure.figsize"] = (8, 6)
    fig, ax = plt.subplots()

    benches = [0.02, 0.12, 0.188, 0.346, 0.9, 2.58, 11.2, 51.6, 320, 1564]
    x_ticks = ["25", "50", "100", "200", "400", "800", "1600", "3200", "6400", "12800"]
    ranger = [25,50,100,200,400, 800, 1600, 3200, 6400, 12800]
    plt.xticks(ranger,x_ticks, rotation=90)

    # ax.set_yscale("log", base=2)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(22))

    # plt.plot(ranger, benches, color="r")
    ax.plot(ranger, benches, color="r")
    # plt.semilogy(ranger,benches, base=2)
    plt.title(f"scipy.matmul with multi runtime: Static P=400, Iters=1000, scaling T")
    plt.ylabel("Time [s]")
    plt.xlabel("T")
    plt.tight_layout()
    plt.show()
    plt.close()


    #
    # ------------------------------
    # Plot BOTH
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["figure.figsize"] = (8, 6)
    fig, ax = plt.subplots()

    benches_no_multi = [0.028, 0.046, 0.124, 0.374, 1.34, 5.74, 24.5, 108.8, 554, 2494]
    benches_with_multi = [0.02, 0.12, 0.188, 0.346, 0.9, 2.58, 11.2, 51.6, 320, 1564]
    x_ticks = ["25", "50", "100", "200", "400", "800", "1600", "3200", "6400", "12800"]
    ranger = [25,50,100,200,400, 800, 1600, 3200, 6400, 12800]
    plt.xticks(ranger,x_ticks, rotation=90)

    ax.set_yscale("log", base=10)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(22))

    # plt.plot(ranger, benches, color="r")
    ax.plot(ranger, benches_with_multi, color="r", label="With multi")
    ax.plot(ranger, benches_no_multi, color="b", label="Without multi")
    # plt.semilogy(ranger,benches, base=2)
    plt.title(f"scipy.matmul runtime: Static P=400, Iters=1000, scaling T")
    plt.ylabel("Time [s]")
    plt.xlabel("T")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_or_bench_convar_runtimes():
    gamma = 0.97
    plot = 2
    # Mid | Full
    """NOT USED"""
    # Using scipy Algos
    # 2000 iters
    # data = np.random.rand(25, 1000)
    # data = np.random.rand(50, 1000)
    # data = np.random.rand(100, 1000)
    # data = np.random.rand(200, 1000)            #
    # data = np.random.rand(400, 1000)            # 0.07 | 17.37
    # data = np.random.rand(800, 1000)            # 0.22 | 41.09
    # data = np.random.rand(1600, 1000)            # 0.85 | 117.61
    # 200 iters
    # data = np.random.rand(3200, 1000)            # 3.4 | 45.66 | from 200 iters, multiplied by 10: (45.66 - 3.4) * 10 =
    # 100 iters
    # data = np.random.rand(6400, 1000)            # 14.39 | 106.41 | from 100 iters, multiplied by 20: (106.41-14.39) * 20 =
    # data = np.random.rand(12800, 1000)            # 66.1 | 110.93 | from 10 iters, multiplied by 200: (110.93-66.1) * 200 =

    # convar.convar_np(data, gamma, 1, early_stop_bool=False, num_iters=10)

    """USED, MOST BENCHES WRITTEN DOWN IN EXCEL: Präsis/03.05/benches.xlsx"""
    # Using scipy Algos
    # 1000 iters
    # data = np.random.rand(25, 400)
    # data = np.random.rand(50, 400)
    # data = np.random.rand(100, 400)
    # data = np.random.rand(200, 400)
    # data = np.random.rand(400, 400)
    # data = np.random.rand(800, 400)
    # data = np.random.rand(1600, 400)
    # data = np.random.rand(3200, 400)
    # data = np.random.rand(6400, 400)
    # data = np.random.rand(12800, 400)

    # convar.convar_np(data, gamma, 1, early_stop_bool=False, num_iters=5000)


    # Plot 1
    if(plot == 1):
        plt.rcParams.update({'font.size': 13})
        plt.rcParams["figure.figsize"] = (8, 6)
        fig, ax = plt.subplots()

        benches = [0.132, 0.534, 0.906, 1.432, 3.48, 10.18, 34.6, 132.6, 686, 3710]
        x_ticks = ["25", "50", "100", "200", "400", "800", "1600", "3200", "6400", "12800"]
        ranger = [25,50,100,200,400, 800, 1600, 3200, 6400, 12800]
        plt.xticks(ranger,x_ticks, rotation=90)

        # ax.set_yscale("log", base=2)
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(22))

        # plt.plot(ranger, benches, color="r")
        ax.plot(ranger, benches, color="r")
        # plt.semilogy(ranger,benches, base=2)
        plt.title(f"Regular Convar scipy runtime: Static P=400, Iters=1000, scaling T")
        plt.ylabel("Time [s]")
        plt.xlabel("T")
        plt.tight_layout()
        plt.show()
        plt.close()


    # Using convar_slim, 1 thread
    # 1000 iters
    # data = np.random.rand(25, 400)                          # 0 | 0.7 | from 5000
    # data = np.random.rand(50, 400)                        # 0 | 1.34 | from 5000
    # data = np.random.rand(100, 400)                     # 0 | 2.76 | from 5000
    # data = np.random.rand(200, 400)                   # 0.01 | 0.73 | from 500 iters
    # data = np.random.rand(400, 400)                     # 0.04 | 2.55 | from 500 iters
    # data = np.random.rand(800, 400)                     # 0.16 | 10.02 | from 500 iters
    # data = np.random.rand(1600, 400)                    # 0.69 | 8.12 | from 100 iters, multiplied by 10: (8.12 - 0.69) * 10 =
    # data = np.random.rand(3200, 400)                    # 2.83 | 32.78 | from 100 iters, multiplied by 20: (32.78 - 2.83) * 10 =
    # data = np.random.rand(6400, 400)                  # 12.38 | 19.23 | from 5 iters, multiplied by 200: (19.23 - 12.38) * 200 =
    # data = np.random.rand(12800, 400)                 # 58.72 | 90.3 | from 5 iters, multiplied by 200: (90.3 - 58.72) * 200 =
    # convar_slim(data, gamma, 1, num_iters=5000)

    # Plot 2 Convar slim, no MP
    if(plot==2):
        # plt.rcParams.update({'font.size': 13})
        # plt.rcParams["figure.figsize"] = (8, 6)
        # fig, ax = plt.subplots()
        #
        # benches = [0.14, 0.268, 0.552, 1.44, 5.02, 19.72, 74.3, 299.5, 1370, 6316]
        # x_ticks = ["", "", "", "", "400", "800", "1600", "3200", "6400", "12800"]
        # ranger = [25,50,100,200,400, 800, 1600, 3200, 6400, 12800]
        # plt.xticks(ranger,x_ticks, rotation=90)
        #
        # ax.set_yscale("log", base=10)
        # ax.set_xscale("log", base=10)
        # # ax.yaxis.set_major_locator(ticker.MultipleLocator(22))
        #
        # # Approximate chunk time
        # chunk_approx_400 = [None, None, None, None, 5.02, 5.02*2, 5.02*4, 5.02*8, 5.02*16, 5.02*32]
        # chunk_approx_1600 = [None, None, None, None, None, None, 74.3, 74.3*2, 74.3*4, 74.3*8]
        #
        # # plt.plot(ranger, benches, color="r")
        # ax.plot(ranger, benches, label="Regular")
        # ax.plot(ranger, chunk_approx_1600,  label="T=1600 chunks")
        # ax.plot(ranger, chunk_approx_400, label="T=400 chunks")
        #
        # # plt.semilogy(ranger,benches, base=2)
        # plt.title(f"Convar (no MP): \nP=400, Iters=1000")
        # # plt.title(f"Convar scipy no multi, slim runtime: Static P=400, Iters=1000, scaling T")
        # plt.ylabel("Time [s]")
        # plt.xlabel("T")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        # plt.close()
        p="loglog"
        # p="both"


    if(plot == 3):
        # Plot 3 both
        plt.rcParams.update({'font.size': 13})
        plt.rcParams["figure.figsize"] = (8, 6)
        fig, ax = plt.subplots()

        benches_reg = [0.132, 0.534, 0.906, 1.432, 3.48, 10.18, 34.6, 132.6, 686, 3710]
        benches_slim = [0.14, 0.268, 0.552, 1.44, 5.02, 19.72, 74.3, 299.5, 1370, 6316]
        x_ticks = ["25", "50", "100", "200", "400", "800", "1600", "3200", "6400", "12800"]
        ranger = [25,50,100,200,400, 800, 1600, 3200, 6400, 12800]
        plt.xticks(ranger,x_ticks, rotation=90)

        ax.set_yscale("log", base=10)
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(22))

        ax.plot(ranger, benches_reg, color="r", label="Regular")
        ax.plot(ranger, benches_slim, color="b", label="No multi, slim")
        # plt.semilogy(ranger,benches, base=2)
        plt.title(f"Convar scipy: Static P=400, Iters=1000, scaling T")
        plt.ylabel("Time [s]")
        plt.xlabel("T")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()


    # Using convar_own
    # 200 iters
    # data = np.random.rand(20, 30)           # 0.01 | 3.69
    # data = np.random.rand(40, 30)           # 0.06 | 14.29
    # data = np.random.rand(80, 30)           # 0.34 | 56.95
    # convar_own_matmul(data, gamma, 1, num_iters=200)

def bachelor_chunking():
    """ BACHELOR THESIS VERSION """
    """ PLOT 1 """
    # plt.rcParams.update({'font.size': 14})
    # plt.rcParams["figure.figsize"] = (6, 5)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["figure.figsize"] = (6.5 * 0.6, 6.5 * 0.75 * 0.6)  # Bachelor Thesis page width and 4.32=good looking
    fig, ax = plt.subplots()

    benches = [0.14, 0.268, 0.552, 1.44, 5.02, 19.72, 74.3, 299.5, 1370, 6316]
    x_ticks = ["", "", "", "", "400", "800", "1600", "3200", "6400", "12800"]
    ranger = [25,50,100,200,400, 800, 1600, 3200, 6400, 12800]
    # plt.xticks(ranger,x_ticks, rotation=90)

    ax.set_yscale("log", base=10)
    ax.set_xscale("log", base=10)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(22))

    # Approximate chunk time
    chunk_approx_400 = [None, None, None, None, 5.02, 5.02*2, 5.02*4, 5.02*8, 5.02*16, 5.02*32]
    chunk_approx_1600 = [None, None, None, None, None, None, 74.3, 74.3*2, 74.3*4, 74.3*8]

    # plt.plot(ranger, benches, color="r")
    ax.plot(ranger, benches, label="Regular")
    # ax.plot(ranger, chunk_approx_1600,  label="T=1600 chunks")
    # ax.plot(ranger, chunk_approx_400, label="T=400 chunks")

    # plt.semilogy(ranger,benches, base=2)
    # plt.title(f"Convar (no MP): \nP=400, Iters=1000")
    # plt.title(f"Convar scipy no multi, slim runtime: Static P=400, Iters=1000, scaling T")
    plt.ylabel("Time [s]")
    plt.xlabel("$T$")
    # plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(r"F:\Uni Goethe\Informatik\BA\Latex\figure\chunking_loglog.pgf")
    plt.close()

    """ PLOT 2 """
    # plt.rcParams.update({'font.size': 14})
    # plt.rcParams["figure.figsize"] = (6, 5)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["figure.figsize"] = (6.5 * 0.6, 6.5 * 0.75 * 0.6)  # Bachelor Thesis page width and 4.32=good looking
    fig, ax = plt.subplots()

    benches = [0.14, 0.268, 0.552, 1.44, 5.02, 19.72, 74.3, 299.5, 1370, 6316]
    x_ticks = ["", "", "", "", "", "800", "1600", "3200", "6400", "12800"]
    ranger = [25,50,100,200,400, 800, 1600, 3200, 6400, 12800]
    plt.xticks(ranger,x_ticks, rotation=90)

    # ax.set_yscale("log", base=10)
    # ax.set_xscale("log", base=10)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(22))

    # Approximate chunk time
    chunk_approx_400 = [None, None, None, None, 5.02, 5.02*2, 5.02*4, 5.02*8, 5.02*16, 5.02*32]
    chunk_approx_1600 = [None, None, None, None, None, None, 74.3, 74.3*2, 74.3*4, 74.3*8]

    # plt.plot(ranger, benches, color="r")
    ax.plot(ranger, benches, label="No chunks")
    ax.plot(ranger, chunk_approx_1600,  label="$T=1600$ chunks")
    ax.plot(ranger, chunk_approx_400, label="$T=400$ chunks")

    # plt.title(f"Convar (no MP): \nP=400, Iters=1000")
    # plt.title(f"Convar scipy no multi, slim runtime: Static P=400, Iters=1000, scaling T")
    plt.ylabel("Time [s]")
    plt.xlabel("$T$")
    plt.legend()
    plt.tight_layout(pad=0)
    # plt.show()
    plt.savefig(r"F:\Uni Goethe\Informatik\BA\Latex\figure\chunking_comparison.pgf")
    plt.close()