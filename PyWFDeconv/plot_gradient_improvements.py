import numpy as np
import PyWFDeconv as wfd
from scipy.io import loadmat
import datetime
from . import convar as convar
from . import early_stops as early_stops
import matplotlib.pyplot as plt
from . import helpers






def show_plots(plotnr):
    #Merav Data
    merav_data_path = r"ExampleData/Clancy_etal_fluorescence_example.mat"
    #Jonas Data
    jonas_data_path = r"ExampleData/F0205_tseries_38_DF_by_F0_351_50_cut200_600_compressed.npz"

    npz_file = np.load(jonas_data_path)
    jonas = npz_file["data"]

    data_import = loadmat(merav_data_path)
    merav = data_import["cal_data"]

    if(plotnr==0):
        data = merav
        # data = data[0::2]       # Odd traces
        gamma = 0.97
        best_lambda = 1
        num_iters = 10000

    if(plotnr==1):
        # print(np.min(jonas))
        jonas = helpers.normalize_1_0(jonas)
        # print(np.min(jonas))
        # data = jonas[:200, :500]
        data = jonas
        gamma = 0.92
        # best_lambda = 3
        best_lambda = 4
        # num_iters= 2000
        num_iters= 10000


    what_to_show = [1,1,1,1]
    early_stop = True


    # Plot 1
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.axhline(y=0.000001, color="r", label="Early Stop Threshold", linestyle=":")

    if(what_to_show[0] == 1):
        r_vanilla, _, _, grad_vanilla = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                         early_stop_bool=early_stop, num_iters=num_iters,
                                                         init_out_matrix_method="rand")
        plt.plot(grad_vanilla, label="Random Init")

    if(what_to_show[1] == 1):
        r_firdif, _, _, grad_firdif = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                       early_stop_bool=early_stop, num_iters=num_iters, adapt_lr_bool=False)
        plt.plot(grad_firdif, label="FirDif Init")

    if(what_to_show[2] == 1):
        r_adaptlr, _, _, grad_adaptlr = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                         early_stop_bool=early_stop, num_iters=num_iters, adapt_lr_bool=True,
                                                         instant_lr_boost=False)
        plt.plot(grad_adaptlr, label="Firdif Init + AdaptLR")

    if(what_to_show[3] == 1):
        r_instant_adaptlr, _, _, grad_instant_adaptlr = convar.convar_np(data, gamma, best_lambda,
                                                                         return_metric_gradients=True,
                                                                         early_stop_bool=early_stop, num_iters=num_iters,
                                                                         adapt_lr_bool=True, instant_lr_boost=True)
        plt.plot(grad_instant_adaptlr, label="Firdif Init + AdaptLR\n + Instant Increase", color="m")

    plt.ylabel("metric(gradient)")
    plt.xlabel("Iteration")
    plt.title("Convar: metric(gradient) comparison")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()




def divergence_example(plotnr):
    #Merav Data
    merav_data_path = r"ExampleData/Clancy_etal_fluorescence_example.mat"
    #Jonas Data
    jonas_data_path = r"ExampleData/F0205_tseries_38_DF_by_F0_351_50_cut200_600_compressed.npz"

    npz_file = np.load(jonas_data_path)
    jonas = npz_file["data"]

    data_import = loadmat(merav_data_path)
    merav = data_import["cal_data"]

    if(plotnr==0):
        data = merav
        # data = data[0::2]       # Odd traces
        gamma = 0.97
        best_lambda = 1
        num_iters = 80

    if(plotnr==1):
        # print(np.min(jonas))
        jonas = helpers.normalize_1_0(jonas)
        # print(np.min(jonas))
        data = jonas[:200, :500]
        gamma = 0.92
        best_lambda = 3
        num_iters= 2000


    what_to_show = [1,0,0,0]


    # Plot 1
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.axhline(y=0.000001, color="r", label="Early Stop Threshold", linestyle=":")


    # IMPORTANT: Tune s increases and decreases in adaptLR such that divergence happens
    # For example: for plotnr=0, s*=4 and s*=0.9
    r_adaptlr, _, _, grad_adaptlr = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                     early_stop_bool=False, num_iters=num_iters, adapt_lr_bool=True,
                                                     instant_lr_boost=False)
    plt.plot(grad_adaptlr, label="Firdif Init + AdaptLR")

    plt.ylabel("metric(gradient)")
    plt.xlabel("Iteration")
    plt.title("Divergence + Overflow")
    plt.yscale("log")
    # plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


