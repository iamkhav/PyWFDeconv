import numpy as np
import time
from pytictoc import TicToc


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
    r = np.ones((np.shape(y)[0], np.shape(y)[1]))

    mid = time.time()
    t = TicToc()
    lineperline_time = [0,0,0,0,0,0]
    for i in range(0, 10000):
        t.tic()
        Ar = A @ r
        lineperline_time[0] += t.tocvalue(restart=True)
        tmAr = (tildey - Ar)
        lineperline_time[1] += t.tocvalue(restart=True)
        At_tmAr = np.transpose(A) @ tmAr
        lineperline_time[2] += t.tocvalue(restart=True)
        Zr = Z @ r
        lineperline_time[3] += t.tocvalue(restart=True)
        x = r + s*At_tmAr - s*_lambda*Zr
        lineperline_time[4] += t.tocvalue(restart=True)
        r = x
        r[r < 0] = 0
        lineperline_time[5] += t.tocvalue(restart=True)
        r[0] = x[0]
    r_final = r[1:]
    r1 = r[0:1]
    beta_0 = np.mean(y - (Dinv @ r), axis=0)
    print(lineperline_time)
    print("------------------------------------------------------")
    print("Numpy stats")
    print("Mid convar time:", mid-start)
    print("Convar time:", time.time()-start)

    # BENCH 400x50
    # lineperline_time print [4.142800200000016, 0.2378835999999469, 4.282995200000019, 4.162444799999945, 0.6910463999998777, 0.1563436000000289]
    # Mid convar time: 0.04153633117675781
    # Convar time: 13.760328769683838

    # Matmuls 4.143+4.283+4.163 = 12.589
    # Matinv 0.04
    # Rest of main loop ops 0.238 + 0.691 + 0.1563

    # BENCH 400x200
    # lineperline_time print [5.066088800000193, 0.7562708000000868, 5.242963800000015, 5.029015400000036, 1.9040692999999884, 0.38031949999993464]
    # Mid convar time: 0.04103565216064453
    # Convar time: 18.459869384765625

    # Matmuls 5.066+5.243+5.029 = 15.338
    # Matinv 0.04
    # Rest of main loop ops 0.756 + 1.9 + 0.38 = 3.036

    return r_final,r1,beta_0
