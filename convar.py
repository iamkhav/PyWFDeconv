import numpy as np
from scipy import linalg
import torch
import time
# import gc
from numba import jit




def convar_np(y, gamma, _lambda):
    """
    convar is a straight translation from matlab into numpy.
    Performance is about 2.5x worse than the matlab function.
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
    # initializing
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = np.ones((np.shape(y)[0], np.shape(y)[1]))  # Test line for consistency instead of randomness

    mid = time.time()

    for i in range(0, 10000):
        Ar = np.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = np.matmul(np.transpose(A), tmAr)
        Zr = np.matmul(Z, r)
        x = r + s*At_tmAr - s*_lambda*Zr
        r = x
        r[r < 0] = 0
        r[0] = x[0]

        # #Early Stoppage
        # gradient = s * At_tmAr - s * _lambda * Zr
        # print("-----------------------")
        # print("Mean:", np.mean(gradient))
        # print("Max:", np.max(gradient))
        # print("Min:", np.min(gradient))
        # print("Sum Max Min:", np.abs(np.max(gradient)) + np.abs(np.min(gradient)))

    r_final = r[1:]
    r1 = r[0]
    beta_0 = np.mean(y - np.matmul(Dinv, r), axis=0)

    print("------------------------------------------------------")
    print("Numpy stats")
    print("Mid convar time:", mid-start)
    print("Convar time:", time.time()-start)

    return r_final,r1,beta_0

def convar_np_at(y, gamma, _lambda):
    """
    This version uses @ instead of np.matmul()
    For some reason, it seems to be slightly faster sometimes.
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
    r = np.ones((np.shape(y)[0], np.shape(y)[1]))  # Test line for consistency instead of randomness

    mid = time.time()

    for i in range(0, 10000):
        Ar = A @ r
        tmAr = (tildey - Ar)
        At_tmAr = np.transpose(A) @ tmAr
        Zr = Z @ r
        x = r + s*At_tmAr - s*_lambda*Zr
        r = x
        r[r < 0] = 0
        r[0] = x[0]
    r_final = r[1:]
    r1 = r[0]
    beta_0 = np.mean(y - (Dinv @ r), axis=0)

    print("------------------------------------------------------")
    print("Numpy stats")
    print("Mid convar time:", mid-start)
    print("Convar time:", time.time()-start)

    return r_final,r1,beta_0


@jit(nopython=True)
def convar_numba(y, gamma, _lambda):
    """
    Don't use this function.
    It doesn't return correct values due to having certain numpy functions not supported by numba.
    -Amon

    :param y:
    :param gamma:
    :param _lambda:
    :return:
    """
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

    Z = np.transpose(L1) @ L1

    # large step size that ensures converges
    s = 0.5 * ( (1-gamma)**2 / ( (1-gamma**T)**2 + (1-gamma)**2 * 4 * _lambda ) )

    # deconvolution
    # initializing
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = np.ones((np.shape(y)[0], np.shape(y)[1]))  # Test line for consistency instead of randomness


    for i in range(0, 10000):
        Ar = A @ r
        tmAr = (tildey - Ar)
        At_tmAr = np.transpose(A) @ tmAr
        Zr = Z @ r
        x = r + s*At_tmAr - s*_lambda*Zr
        r = x
        # r[r < 0] = 0
        r[0] = x[0]
    r_final = r[1:]
    r1 = r[0]
    beta_0 = np.mean(y - Dinv @ r)

    print("------------------------------------------------------")
    print("Numpy stats")

    return r_final,r1,beta_0

def convar_np_1line(y, gamma, _lambda):
    """
    convar is a straight translation from matlab into numpy.
    Performance is about 2.5x worse than the matlab function.
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
    # initializing
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = np.ones((np.shape(y)[0], np.shape(y)[1]))  # Test line for consistency instead of randomness

    mid = time.time()

    for i in range(0, 10000):
        x = r + s*(np.matmul(np.transpose(A), (tildey - np.matmul(A, r)))) - s*_lambda*(np.matmul(Z, r))
        r = x
        r[r < 0] = 0
        r[0] = x[0]
    r_final = r[1:]
    r1 = r[0]
    beta_0 = np.mean(y - np.matmul(Dinv, r), axis=0)

    print("------------------------------------------------------")
    print("Numpy 1Line stats")
    print("Mid convar time:", mid-start)
    print("Convar time:", time.time()-start)

    return r_final,r1,beta_0


def convar_np_F_dot(y, gamma, _lambda):
    """
    convar is a straight translation from matlab into numpy.
    Performance is about 2.5x worse than the matlab function.
    -Amon

    :param y:
    :param gamma:
    :param _lambda:
    :return:
    """
    start = time.time()
    y = np.copy(y, order="F")

    T = np.shape(y)[0]
    P = np.identity(T) - 1 / T * np.ones((T,T))
    P = np.copy(P, order="F")
    tildey = np.matmul(P, y)

    # will be used later to reconstruct the calcium from the deconvoled rates
    Dinv = np.zeros((T, T), order="F")

    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp

    A = np.dot(P, Dinv)

    L1 = np.zeros((T, T), order="F")
    for i in range(0, T):
        for j in range(0, T):
            if(i >= 2 and j >= 1):
                if(i == j):
                    L1[i][j] = 1
                if(i == j+1):
                    L1[i][j] = -1

    Z = np.dot(np.transpose(L1), L1)

    # large step size that ensures converges
    s = 0.5 * ( (1-gamma)**2 / ( (1-gamma**T)**2 + (1-gamma)**2 * 4 * _lambda ) )

    # deconvolution
    # initializing
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = np.ones((np.shape(y)[0], np.shape(y)[1]), order="F")  # Test line for consistency instead of randomness

    mid = time.time()

    for i in range(0, 10000):
        Ar = np.dot(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = np.dot(np.transpose(A), tmAr)
        Zr = np.dot(Z, r)
        x = r + s*At_tmAr - s*_lambda*Zr
        r = x
        r[r < 0] = 0
        r[0] = x[0]
    r_final = r[1:]
    r1 = r[0]
    beta_0 = np.mean(y - np.dot(Dinv, r), axis=0)

    print("------------------------------------------------------")
    print("Numpy F Dot stats")
    print("Mid convar time:", mid-start)
    print("Convar time:", time.time()-start)

    return r_final,r1,beta_0

def convar_np_openblas(y, gamma, _lambda):
    """
    convar is a straight translation from matlab into numpy.
    Performance is about 2.5x worse than the matlab function.
    -Amon

    :param y:
    :param gamma:
    :param _lambda:
    :return:
    """
    start = time.time()
    y = np.copy(y, order="F")

    T = np.shape(y)[0]
    P = np.identity(T) - 1 / T * np.ones((T,T))
    P = np.copy(P, order="F")
    tildey = linalg.blas.sgemm(1, P, y)

    # will be used later to reconstruct the calcium from the deconvoled rates
    Dinv = np.zeros((T, T), order="F")

    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp

    A = linalg.blas.sgemm(1, P, Dinv)

    L1 = np.zeros((T, T), order="F")
    for i in range(0, T):
        for j in range(0, T):
            if(i >= 2 and j >= 1):
                if(i == j):
                    L1[i][j] = 1
                if(i == j+1):
                    L1[i][j] = -1


    Z = linalg.blas.sgemm(1, np.transpose(L1), L1)

    # large step size that ensures converges
    s = 0.5 * ( (1-gamma)**2 / ( (1-gamma**T)**2 + (1-gamma)**2 * 4 * _lambda ) )

    # deconvolution
    # initializing
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = np.ones((np.shape(y)[0], np.shape(y)[1]), order="F")  # Test line for consistency instead of randomness

    mid = time.time()



    for i in range(0, 10000):
        Ar = linalg.blas.sgemm(1, A, r)
        tmAr = (tildey - Ar)
        At_tmAr = linalg.blas.sgemm(1, np.transpose(A), tmAr)
        Zr = linalg.blas.sgemm(1, Z, r)
        x = r + s*At_tmAr - s*_lambda*Zr
        r = x
        r[r < 0] = 0
        r[0] = x[0]


    r_final = r[1:]
    r1 = r[0]
    beta_0 = np.mean(y - linalg.blas.sgemm(1, Dinv, r), axis=0)

    print("------------------------------------------------------")
    print("OpenBLAS stats")
    print("Mid convar time:", mid-start)
    print("Convar time:", time.time()-start)

    return r_final,r1,beta_0


def convar_torch(y, gamma, _lambda):
    """
    convar_torch implements torch data structures and uses the CPU.
    -Amon

    :param y:
    :param gamma:
    :param _lambda:
    :return:
    """
    # INIT
    y = torch.from_numpy(y)
    start = time.time()


    T = y.shape[0]
    P = torch.eye(T) - 1 / T * torch.ones((T, T))
    tildey = torch.matmul(P, y)

    # will be used later to reconstruct the calcium from the deconvoled rates
    Dinv = torch.zeros((T, T))

    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp

    A = torch.matmul(P, Dinv)

    L1 = torch.zeros((T, T))
    for i in range(0, T):
        for j in range(0, T):
            if (i >= 2 and j >= 1):
                if (i == j):
                    L1[i][j] = 1
                if (i == j + 1):
                    L1[i][j] = -1

    Z = torch.matmul(torch.transpose(L1, 0, 1), L1)

    # large step size that ensures converges
    s = 0.5 * ((1 - gamma) ** 2 / ((1 - gamma ** T) ** 2 + (1 - gamma) ** 2 * 4 * _lambda))

    # deconvolution
    # initializing
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = torch.ones((y.shape[0], y.shape[1]))  # Test line for consistency instead of randomness

    mid = time.time()
    # All code until here is very light

    for i in range(0, 10000):
        Ar = torch.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = torch.matmul(torch.transpose(A, 0, 1), tmAr)
        Zr = torch.matmul(Z, r)
        x = r + s * At_tmAr - s * _lambda * Zr
        r = x
        r[r < 0] = 0
        r[0] = x[0]

    r_final = r[1:]
    r1 = r[0]
    beta_0 = torch.mean(y - torch.matmul(Dinv, r), dim=0)

    print("------------------------------------------------------")
    print("Torch stats")
    print("Mid convar time:", mid - start)
    print("Convar time:", time.time() - start)

    return r_final.numpy(),r1.numpy(),beta_0.numpy()


def convar_torch_cuda(y, gamma, _lambda):
    """
    convar_torch implements torch data structures in order to use CUDA.
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
    y = torch.from_numpy(y).to(device)
    start = time.time()


    T = y.shape[0]
    P = torch.eye(T) - 1 / T * torch.ones((T, T))
    P = P.to(device)
    tildey = torch.matmul(P, y)

    # will be used later to reconstruct the calcium from the deconvoled rates
    Dinv = (torch.zeros((T, T))).to(device)

    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp

    A = torch.matmul(P, Dinv)

    L1 = torch.zeros((T, T))
    L1 = L1.to(device)
    for i in range(0, T):
        for j in range(0, T):
            if (i >= 2 and j >= 1):
                if (i == j):
                    L1[i][j] = 1
                if (i == j + 1):
                    L1[i][j] = -1

    Z = torch.matmul(torch.transpose(L1, 0, 1), L1)

    # large step size that ensures converges
    s = 0.5 * ((1 - gamma) ** 2 / ((1 - gamma ** T) ** 2 + (1 - gamma) ** 2 * 4 * _lambda))

    # deconvolution
    # initializing
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = (torch.ones((y.shape[0], y.shape[1]))).to(device)  # Test line for consistency instead of randomness

    mid = time.time()
    # All code until here is very light

    for i in range(0, 10000):
        Ar = torch.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = torch.matmul(torch.transpose(A, 0, 1), tmAr)
        Zr = torch.matmul(Z, r)
        x = r + s * At_tmAr - s * _lambda * Zr
        r = x
        r[r < 0] = 0
        r[0] = x[0]

    r_final = r[1:]
    r1 = r[0]
    beta_0 = torch.mean(y - torch.matmul(Dinv, r), dim=0)


    print("------------------------------------------------------")
    print("Torch CUDA stats")
    print("Mid convar time:", mid - start)
    print("Convar time:", time.time() - start)

    return r_final.cpu().numpy(),r1.cpu().numpy(),beta_0.cpu().numpy()


def convar_torch_cuda_direct(y, gamma, _lambda):
    """
    convar_torch implements torch data structures in order to use CUDA.
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
    y = torch.from_numpy(y).to(device)
    start = time.time()


    T = y.shape[0]
    P = torch.eye(T) - 1 / T * torch.ones((T, T))
    P = P.to(device)
    tildey = torch.matmul(P, y)

    # will be used later to reconstruct the calcium from the deconvoled rates
    Dinv = torch.empty((T,T), device=device)
    Dinv = torch.zeros_like(Dinv)

    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp

    A = torch.matmul(P, Dinv)

    L1 = torch.empty((T,T), device=device)
    L1 = torch.zeros_like(L1)
    for i in range(0, T):
        for j in range(0, T):
            if (i >= 2 and j >= 1):
                if (i == j):
                    L1[i][j] = 1
                if (i == j + 1):
                    L1[i][j] = -1

    Z = torch.matmul(torch.transpose(L1, 0, 1), L1)

    # large step size that ensures converges
    s = 0.5 * ((1 - gamma) ** 2 / ((1 - gamma ** T) ** 2 + (1 - gamma) ** 2 * 4 * _lambda))

    # deconvolution
    # initializing
    # r = np.random.rand(y.shape[0], y.shape[1])
    r = torch.empty((y.shape[0], y.shape[1]), device=device)
    r = torch.ones_like(r)  # Test line for consistency instead of randomness

    mid = time.time()
    # All code until here is very light

    for i in range(0, 10000):
        Ar = torch.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = torch.matmul(torch.transpose(A, 0, 1), tmAr)
        Zr = torch.matmul(Z, r)
        x = r + s * At_tmAr - s * _lambda * Zr
        r = x
        r[r < 0] = 0
        r[0] = x[0]

    r_final = r[1:]
    r1 = r[0]
    beta_0 = torch.mean(y - torch.matmul(Dinv, r), dim=0)


    print("------------------------------------------------------")
    print("Torch CUDA Direct stats")
    print("Mid convar time:", mid - start)
    print("Convar time:", time.time() - start)

    return r_final.cpu().numpy(),r1.cpu().numpy(),beta_0.cpu().numpy()

def convar_half_torch(y, gamma, _lambda):
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

    # deconvolution
    # initializing
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = np.ones((np.shape(y)[0], np.shape(y)[1]))  # Test line for consistency instead of randomness

    mid = time.time()

    # Torch allocation
    A = torch.from_numpy(A)
    r = torch.from_numpy(r)
    tildey = torch.from_numpy(tildey)
    Z = torch.from_numpy(Z)
    Dinv = torch.from_numpy(Dinv)
    y = torch.from_numpy(y)

    for i in range(0, 10000):
        Ar = torch.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = torch.matmul(torch.transpose(A, 0, 1), tmAr)
        Zr = torch.matmul(Z, r)
        x = r + s * At_tmAr - s * _lambda * Zr
        r = x
        r[r < 0] = 0
        r[0] = x[0]

    r_final = r[1:]
    r1 = r[0]
    beta_0 = torch.mean(y - torch.matmul(Dinv, r), dim=0)

    print("------------------------------------------------------")
    print("Half Torch stats")
    print("Mid convar time:", mid-start)
    print("Convar time:", time.time()-start)

    return r_final.numpy(),r1.numpy(),beta_0.numpy()


# @torch.jit.script
def convar_half_torch_jit(y, gamma, _lambda):
    """
    Using numpy to initialize the matrices, then converting them into pytorch tensors.
    Returning Numpy arrays at the end.
    -Amon

    :param y:
    :param gamma:
    :param _lambda:
    :return:
    """

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
    # initializing
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = np.ones((np.shape(y)[0], np.shape(y)[1]))  # Test line for consistency instead of randomness


    # Torch allocation
    A = torch.from_numpy(A)
    r = torch.from_numpy(r)
    tildey = torch.from_numpy(tildey)
    Z = torch.from_numpy(Z)
    Dinv = torch.from_numpy(Dinv)
    y = torch.from_numpy(y)

    for i in range(0, 10000):
        Ar = torch.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = torch.matmul(torch.transpose(A, 0, 1), tmAr)
        Zr = torch.matmul(Z, r)
        x = r + s * At_tmAr - s * _lambda * Zr
        r = x
        r[r < 0] = 0
        r[0] = x[0]

    r_final = r[1:]
    r1 = r[0]
    beta_0 = torch.mean(y - torch.matmul(Dinv, r), dim=0)

    print("------------------------------------------------------")
    print("Half Torch stats")


    return r_final.numpy(),r1.numpy(),beta_0.numpy()

@torch.jit.script
def convar_torch_jit(y, gamma, _lambda):
    """
    convar_torch implements torch data structures and uses the CPU.
    -Amon

    :param y:
    :param gamma:
    :param _lambda:
    :return:
    """

    T = y.shape[0]
    P = torch.eye(T) - 1 / T * torch.ones((T, T))
    tildey = torch.matmul(P, y)

    # will be used later to reconstruct the calcium from the deconvoled rates
    Dinv = torch.zeros((T, T))

    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp

    A = torch.matmul(P, Dinv)

    L1 = torch.zeros((T, T))
    for i in range(0, T):
        for j in range(0, T):
            if (i >= 2 and j >= 1):
                if (i == j):
                    L1[i][j] = 1
                if (i == j + 1):
                    L1[i][j] = -1

    Z = torch.matmul(torch.transpose(L1, 0, 1), L1)

    # large step size that ensures converges
    s = 0.5 * ((1 - gamma) ** 2 / ((1 - gamma ** T) ** 2 + (1 - gamma) ** 2 * 4 * _lambda))

    # deconvolution
    # initializing
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = torch.ones((y.shape[0], y.shape[1]))  # Test line for consistency instead of randomness

    # All code until here is very light

    for i in range(0, 10000):
        Ar = torch.matmul(A, r)
        tmAr = (tildey - Ar)
        At_tmAr = torch.matmul(torch.transpose(A, 0, 1), tmAr)
        Zr = torch.matmul(Z, r)
        x = r + s * At_tmAr - s * _lambda * Zr
        r = x
        r[r < 0] = 0
        r[0] = x[0]

    r_final = r[1:]
    r1 = r[0]
    beta_0 = torch.mean(y - torch.matmul(Dinv, r), dim=0)

    print("------------------------------------------------------")
    print("Torch stats")


    return r_final,r1,beta_0
