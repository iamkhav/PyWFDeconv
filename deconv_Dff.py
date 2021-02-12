from import_library import *

# from torch.multiprocessing import Process

def deconv(data_path):
    # Load data
    paper_data_import = loadmat(data_path)
    cal_data = paper_data_import["cal_data"]

    # a more convenient (and faster) scaling to work with
    cal_data = cal_data * 100

    # calcium decay rate (single neuron, based on 40Hz mesearmunts in Gcamp6f mice)
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
    all_lambda = [80, 40, 20, 10, 7, 5, 3, 2, 1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    # all_lambda = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]         # Debugging array

    # will be used later to reconstruct the calcium from the deconvoled rates
    Dinv = np.zeros((T, T))
    insert_vec = 1  # This line is not used

    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp

    # saving the results
    # here the penalty (l2) is the same as the fluctuations (l2)
    penalty_size_convar = np.zeros((len(all_lambda), rep))
    calcium_dif_convar = np.zeros((len(all_lambda), rep))

    start = time.time()
    delete_me = []
    for k in range(0, len(all_lambda)):
        _lambda = all_lambda[k]
        # r, r1, beta0 = convar.convar_half_torch(odd_traces, gamma, _lambda)
        # calculating the changes in spiking rate in each deconvolve trace
        # Done token


    # partial_f = partial(convar.convar_half_torch, odd_traces, gamma)
    # with Pool(2) as p:
    #     p.map(partial_f, all_lambda)



    #     p = Process(target=convar.convar_half_torch, args=(odd_traces, gamma, _lambda))
    #     # Process(target=convar.convar_np, args=(odd_traces, gamma, _lambda)).start()
    #     delete_me.append(p)
    #     p.start()
    # for d in delete_me:
    #     d.join()
    end = time.time()
    print(end - start)

    # r, r1, beta0 = convar.convar_half_torch(odd_traces, gamma, all_lambda[0])  # Right now, using numpy to initialize the data and then torch for the big matrix ops is quickest
    # r, r1, beta1 = convar.convar_torch(odd_traces, gamma, all_lambda[0])
    # r, r1, beta2 = convar.convar_torch_cuda_direct(odd_traces, gamma, all_lambda[0])
    # r, r1, beta3 = convar.convar_torch_cuda(odd_traces, gamma, all_lambda[0])
    r, r1, beta4 = convar.convar_np_openblas(odd_traces, gamma, all_lambda[0])
    # r, r1, beta5 = convar.convar_np(odd_traces, gamma, all_lambda[0])

    # print("0", beta0)
    # print("1", beta1)
    # print("2", beta2)
    # print("3", beta3)
    # print("4", beta4)
    # print("5", beta5)
    print(beta4.dtype)