from . import convar as convar
from . import helpers as helpers
from . import wrappers as wfd
from . import firdif as firdif
import numpy as np
import matplotlib.pyplot as plt
from math import ceil


def plot_r_finals(r_final_list):
    start_Frame = 10

    # Plot 1
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["figure.figsize"] = (8, 6)

    c = 1
    for r in r_final_list:
        mean_per_frame = []
        for i in r[start_Frame:]:
            mean_per_frame.append(np.mean(i))
        mean_per_frame = helpers.normalize_1_0(mean_per_frame)
        plt.plot(mean_per_frame, label=c, linewidth=2)
        c += 1

    plt.ylabel("Mean per Frame")
    plt.xlabel("Frame")
    plt.title("Convar: Mean of Output per Frame")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_every_trace_every_way_and_save(data):
    gamma = 0.92
    early_stop = True
    num_iters = 10000
    es_thresh = 0.000001

    start_Frame = 300
    # end_Frame = 200
    start_Trace = 10
    end_Trace = 20
    which_way = [1,1,1]
    # which_way = [0,1,0]



    lambda_list = wfd.generate_lambda_list(1, 5, 0.1)
    # best_lambda = wfd.find_best_lambda(data[10:, :20], gamma=0.92, convar_num_iters=2000, adapt_lr_bool=True,
    #                                    num_workers=0, binary_seach_find_bool=True, all_lambda=lambda_list)  # Jonas data, Gamma adjusted
    best_lambda = 3

    data = data[:, start_Trace:end_Trace]

    if(which_way[0] == 1):
        r_vanilla, _, _, grad_vanilla = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                         early_stop_bool=early_stop, num_iters=num_iters,
                                                         init_out_matrix_method="rand", early_stop_threshold=es_thresh)
    if(which_way[1] == 1):
        r_firdif, _, _ = firdif.firdif_np(data, gamma)
    if(which_way[2] == 1):
        r_instant_adaptlr, _, _, grad_instant_adaptlr = convar.convar_np(data, gamma, best_lambda,
                                                                     return_metric_gradients=True,
                                                                     early_stop_bool=early_stop, num_iters=num_iters,
                                                                     adapt_lr_bool=True, instant_lr_boost=True, early_stop_threshold=es_thresh)

    # Correct Offset
    data = data[1:]


    suffix = ""
    if (which_way[0] == 1):
        suffix += "_stern"
    if (which_way[1] == 1):
        suffix += "_firdif"
    if (which_way[2] == 1):
        suffix += "_allfeatures"



    for i in range(0, np.shape(data)[1]):
        plt.rcParams.update({'font.size': 13})
        plt.rcParams["figure.figsize"] = (14, 10)
        plt.ylabel("Value")
        plt.xlabel("t")


        plt.plot(helpers.normalize_1_0(data[start_Frame:, i]), label="Fluorescence", linewidth=2, alpha=0.3)
        if (which_way[0] == 1):
            plt.plot(helpers.normalize_1_0(r_vanilla[start_Frame:, i]), label="Stern Version", linewidth=2, color="tab:green")
        if (which_way[1] == 1):
            plt.plot(helpers.normalize_1_0(r_firdif[start_Frame:, i]), label="Firdif", linewidth=2, color="tab:orange")
        if (which_way[2] == 1):
            plt.plot(helpers.normalize_1_0(r_instant_adaptlr[start_Frame:, i]), label="All Features", linewidth=2, color="tab:purple")


        plt.title(f"Convar on Fluorescence trace {start_Trace + i}")
        plt.legend()


        plt.tight_layout()
        # plt.show()
        plt.savefig(rf"Plots/{start_Trace + i}{suffix}")
        plt.close()