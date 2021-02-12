from import_library import *

#Todo
# https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics
# Create tensors on gpu instead of copying them over

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
    r = np.ones((200, 50))  # Test line for consistency instead of randomness

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


    r_final = r[1:]
    r1 = r[0]
    beta_0 = np.mean(y - np.matmul(Dinv, r), axis=0)

    print("------------------------------------------------------")
    print("Numpy stats")
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
    r = torch.ones((200, 50))  # Test line for consistency instead of randomness

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

    return r_final, r1, beta_0


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
    r = (torch.ones((200, 50))).to(device)  # Test line for consistency instead of randomness

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

    return r_final, r1, beta_0


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
    # r = np.random.rand(np.shape(y)[0], np.shape(y)[1])
    r = torch.empty((200, 50), device=device)
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

    return r_final, r1, beta_0


def convar_half_torch(y, gamma, _lambda):
    """
    Using numpy to initialize the matrices, then converting them into pytorch tensors.
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
    r = np.ones((200, 50))  # Test line for consistency instead of randomness

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

    return r_final,r1,beta_0

