import numpy as np
import PyWFDeconv as wfd
from scipy.io import loadmat
import datetime
from . import convar as convar
from . import early_stops as early_stops
import matplotlib.pyplot as plt
from . import helpers
import h5py





def show_plots(phase):
    #Merav Data
    merav_data_path = r"ExampleData/Clancy_etal_fluorescence_example.mat"
    #Jonas Data
    jonas_data_path = r"ExampleData/F0205_tseries_38_DF_by_F0_351_50_cut200_600_compressed.npz"

    npz_file = np.load(jonas_data_path)
    jonas = npz_file["data"]

    data_import = loadmat(merav_data_path)
    merav = data_import["cal_data"]

    if(phase==0 or phase==1 or phase==2):
        data = merav
        # data = np.concatenate((data, data), axis=1)
        # data = np.concatenate((data, data), axis=1)
        # print(np.shape(data))
        # data = data[0::2]       # Odd traces
        gamma = 0.97
        best_lambda = 1
        num_iters = 10000

    if(phase==3):
        # print(np.min(jonas))
        jonas = helpers.normalize_1_0(jonas)
        # print(np.min(jonas))
        data = jonas
        data = data[:400, :10000]
        gamma = 0.92
        # gamma = 0.8
        # data = data[::2, :10000]
        # gamma = 0.846
        best_lambda = 2
        # best_lambda = 4
        # best_lambda = 10
        num_iters= 10000
        # h5_path = r"ExampleData/data_Jonas.hdf5"
        # # Read files
        # with h5py.File(h5_path, "r") as f:
        #     # List all groups
        #     print(f"Keys: {f.keys()}")
        #
        #     # Get data and ROI
        #     df_fo = np.array(f["DF_by_FO"])
        #     ROI = np.array(f["ROI"])
        # # Clean the data (if you suspect or know about Inf/NaN regions)
        # data = helpers.CleanDFFO(df_fo, ROI=ROI)
        # data = data[:400, 100:110, 100]
        # data = helpers.normalize_1_0(data)


    what_to_show = [1,1,1,1]
    early_stop = True
    early_stop_threshold = 0.000001
    # scaler = 0.7
    scaler = 1

    # Plot 0
    if (phase == 0):
        what_to_show = [1, 0, 0, 0]
        plt.rcParams.update({'font.size': 12})
        # plt.rcParams["figure.figsize"] = (8, 6)                   # Powerpoint Maße
        # plt.rcParams["figure.figsize"] = (9*scaler, 6*scaler)     # Bachelor Thesis
        plt.rcParams["figure.figsize"] = (6.5, 4.32)  # Bachelor Thesis page width and 4.32=good looking
        plt.axhline(y=0.000001, color="r", label="Early stopping threshold", linestyle=":")
        early_stop = False
        omega_find = False
        # omega_find = True

        if (what_to_show[0] == 1):
            r_vanilla, _, _, grad_vanilla = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                             early_stop_bool=early_stop, num_iters=num_iters,
                                                             init_out_matrix_method="rand",
                                                             early_stop_threshold=early_stop_threshold)
            plt.plot(grad_vanilla, label="Stern et al.")

        if (what_to_show[1] == 1):
            r_firdif, _, _, grad_firdif = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                           early_stop_bool=early_stop, num_iters=num_iters,
                                                           adapt_lr_bool=False,
                                                           early_stop_threshold=early_stop_threshold,
                                                           firdif_find_best_omega_bool=omega_find)
            plt.plot(grad_firdif, label="Firdif init")

        if (what_to_show[2] == 1):
            r_adaptlr, _, _, grad_adaptlr = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                             early_stop_bool=early_stop, num_iters=num_iters,
                                                             adapt_lr_bool=True,
                                                             instant_lr_boost=False,
                                                             early_stop_threshold=early_stop_threshold,
                                                             firdif_find_best_omega_bool=omega_find)
            plt.plot(grad_adaptlr, label="Firdif init + adaptLR")

        if (what_to_show[3] == 1):
            r_instant_adaptlr, _, _, grad_instant_adaptlr = convar.convar_np(data, gamma, best_lambda,
                                                                             return_metric_gradients=True,
                                                                             early_stop_bool=early_stop,
                                                                             num_iters=num_iters,
                                                                             adapt_lr_bool=True, instant_lr_boost=True,
                                                                             early_stop_threshold=early_stop_threshold,
                                                                             firdif_find_best_omega_bool=omega_find)
            plt.plot(grad_instant_adaptlr, label="Firdif init + adaptLR\n + instant increase", color="m")

        plt.ylabel("$metric(gradient)$")
        plt.xlabel("iteration")
        # plt.title("Convar: metric(gradient) comparison")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        # plt.show()

        plt.savefig(r"F:\Uni Goethe\Informatik\BA\Latex\figure\gradients_phase0.pgf")
        plt.close()

    # Plot 1
    if (phase==1):
        plt.rcParams.update({'font.size': 12})
        # plt.rcParams["figure.figsize"] = (8, 6)                   # Powerpoint Maße
        # plt.rcParams["figure.figsize"] = (9*scaler, 6*scaler)     # Bachelor Thesis
        plt.rcParams["figure.figsize"] = (6.5, 4.32)                # Bachelor Thesis page width and 4.32=good looking
        plt.axhline(y=0.000001, color="r", label="Early stopping threshold", linestyle=":")
        early_stop = False
        omega_find = False
        # omega_find = True

        if(what_to_show[0] == 1):
            r_vanilla, _, _, grad_vanilla = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                             early_stop_bool=early_stop, num_iters=num_iters,
                                                             init_out_matrix_method="rand", early_stop_threshold=early_stop_threshold)
            plt.plot(grad_vanilla, label="Stern et al.")

        if(what_to_show[1] == 1):
            r_firdif, _, _, grad_firdif = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                           early_stop_bool=early_stop, num_iters=num_iters, adapt_lr_bool=False, early_stop_threshold=early_stop_threshold,
                                                           firdif_find_best_omega_bool=omega_find)
            plt.plot(grad_firdif, label="Firdif init")

        if(what_to_show[2] == 1):
            r_adaptlr, _, _, grad_adaptlr = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                             early_stop_bool=early_stop, num_iters=num_iters, adapt_lr_bool=True,
                                                             instant_lr_boost=False, early_stop_threshold=early_stop_threshold,
                                                             firdif_find_best_omega_bool=omega_find)
            plt.plot(grad_adaptlr, label="Firdif init + adaptLR")

        if(what_to_show[3] == 1):
            r_instant_adaptlr, _, _, grad_instant_adaptlr = convar.convar_np(data, gamma, best_lambda,
                                                                             return_metric_gradients=True,
                                                                             early_stop_bool=early_stop, num_iters=num_iters,
                                                                             adapt_lr_bool=True, instant_lr_boost=True, early_stop_threshold=early_stop_threshold,
                                                                             firdif_find_best_omega_bool=omega_find)
            plt.plot(grad_instant_adaptlr, label="Firdif init + adaptLR\n + instant increase", color="m")

        plt.ylabel("$metric(gradient)$")
        plt.xlabel("iteration")
        # plt.title("Convar: metric(gradient) comparison")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        # plt.show()

        # plt.savefig(r"F:\Uni Goethe\Informatik\BA\Latex\figure\gradients_phase1 no omegafind.pgf")
        plt.savefig(r"F:\Uni Goethe\Informatik\BA\Latex\figure\gradients_phase1 with omegafind.pgf")
        plt.close()

    if (phase == 2):
        plt.rcParams.update({'font.size': 12})
        # plt.rcParams["figure.figsize"] = (8, 6)                   # Powerpoint Maße
        # plt.rcParams["figure.figsize"] = (9*scaler, 6*scaler)     # Bachelor Thesis
        plt.rcParams["figure.figsize"] = (6.5, 4.32)  # Bachelor Thesis page width and 4.32=good looking
        plt.axhline(y=0.000001, color="r", label="Early stopping threshold", linestyle=":")
        omega_find = True

        if (what_to_show[0] == 1):
            r_vanilla, _, _, grad_vanilla = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                             early_stop_bool=early_stop, num_iters=num_iters,
                                                             init_out_matrix_method="rand",
                                                             early_stop_threshold=early_stop_threshold)
            plt.plot(grad_vanilla, label="Stern et al.")

        if (what_to_show[1] == 1):
            r_firdif, _, _, grad_firdif = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                           early_stop_bool=early_stop, num_iters=num_iters,
                                                           adapt_lr_bool=False,
                                                           early_stop_threshold=early_stop_threshold, firdif_find_best_omega_bool=omega_find)
            plt.plot(grad_firdif, label="Firdif init")

        if (what_to_show[2] == 1):
            r_adaptlr, _, _, grad_adaptlr = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                             early_stop_bool=early_stop, num_iters=num_iters,
                                                             adapt_lr_bool=True,
                                                             instant_lr_boost=False,
                                                             early_stop_threshold=early_stop_threshold, firdif_find_best_omega_bool=omega_find)
            plt.plot(grad_adaptlr, label="Firdif init + adaptLR")

        if (what_to_show[3] == 1):
            r_instant_adaptlr, _, _, grad_instant_adaptlr = convar.convar_np(data, gamma, best_lambda,
                                                                             return_metric_gradients=True,
                                                                             early_stop_bool=early_stop,
                                                                             num_iters=num_iters,
                                                                             adapt_lr_bool=True, instant_lr_boost=True,
                                                                             early_stop_threshold=early_stop_threshold, firdif_find_best_omega_bool=omega_find)
            plt.plot(grad_instant_adaptlr, label="Firdif init + adaptLR\n + instant increase", color="m")

        plt.ylabel("$metric(gradient)$")
        plt.xlabel("iteration")
        # plt.title("Convar: metric(gradient) comparison")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        # plt.show()

        # plt.savefig(r"F:\Uni Goethe\Informatik\BA\Latex\figure\gradients_phase2 no omegafind.pgf")
        plt.savefig(r"F:\Uni Goethe\Informatik\BA\Latex\figure\gradients_phase2 with omegafind.pgf")
        plt.close()

    if (phase == 3):
        plt.rcParams.update({'font.size': 12})
        # plt.rcParams["figure.figsize"] = (8, 6)                   # Powerpoint Maße
        # plt.rcParams["figure.figsize"] = (9*scaler, 6*scaler)     # Bachelor Thesis
        plt.rcParams["figure.figsize"] = (6.5, 4.32)                # Bachelor Thesis page width and 4.32=good looking
        plt.axhline(y=0.000001, color="r", label="Early stopping threshold", linestyle=":")
        omega_find = True
        early_stop = True

        if (what_to_show[0] == 1):
            r_vanilla, _, _, grad_vanilla = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                             early_stop_bool=early_stop, num_iters=num_iters,
                                                             init_out_matrix_method="rand",
                                                             early_stop_threshold=early_stop_threshold)
            plt.plot(grad_vanilla, label="Stern et al.")

        if (what_to_show[1] == 1):
            r_firdif, _, _, grad_firdif = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                           early_stop_bool=early_stop, num_iters=num_iters,
                                                           adapt_lr_bool=False,
                                                           early_stop_threshold=early_stop_threshold, firdif_find_best_omega_bool=omega_find)
            plt.plot(grad_firdif, label="Firdif init")

        if (what_to_show[2] == 1):
            r_adaptlr, _, _, grad_adaptlr = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                                             early_stop_bool=early_stop, num_iters=num_iters,
                                                             adapt_lr_bool=True,
                                                             instant_lr_boost=False,
                                                             early_stop_threshold=early_stop_threshold, firdif_find_best_omega_bool=omega_find)
            plt.plot(grad_adaptlr, label="Firdif init + adaptLR")

        if (what_to_show[3] == 1):
            r_instant_adaptlr, _, _, grad_instant_adaptlr = convar.convar_np(data, gamma, best_lambda,
                                                                             return_metric_gradients=True,
                                                                             early_stop_bool=early_stop,
                                                                             num_iters=num_iters,
                                                                             adapt_lr_bool=True, instant_lr_boost=True,
                                                                             early_stop_threshold=early_stop_threshold, firdif_find_best_omega_bool=omega_find)
            plt.plot(grad_instant_adaptlr, label="Firdif init + adaptLR\n + instant increase", color="m")

        plt.ylabel("$metric(gradient)$")
        plt.xlabel("iteration")
        # plt.title("Convar: metric(gradient) comparison")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        # plt.show()
        # plt.savefig(r"F:\Uni Goethe\Informatik\BA\Latex\figure\gradients_phase3.pgf")
        plt.close()


def firdif_omegacomparison():
    merav_data_path = r"ExampleData/Clancy_etal_fluorescence_example.mat"
    jonas_data_path = r"ExampleData/F0205_tseries_38_DF_by_F0_351_50_cut200_600_compressed.npz"

    data_import = loadmat(merav_data_path)
    merav = data_import["cal_data"]
    data = merav
    gamma = 0.97
    best_lambda = 1
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["figure.figsize"] = (6.5, 4.32)                # Bachelor Thesis page width and 4.32=good looking

    # npz_file = np.load(jonas_data_path)
    # jonas = npz_file["data"]
    # jonas = helpers.normalize_1_0(jonas)
    # # print(np.min(jonas))
    # data = jonas
    # data = data[:, :50]
    # gamma = 0.92
    # best_lambda = 3

    # print(np.shape(data))

    num_iters = 1500
    early_stop = True
    early_stop_threshold = 0.000001
    plt.axhline(y=0.000001, color="r", label="Early stopping threshold", linestyle=":")

    _, _, _, grad_firdif_3 = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                              early_stop_bool=early_stop, num_iters=num_iters,
                                              adapt_lr_bool=False,
                                              early_stop_threshold=early_stop_threshold,
                                              firdif_omega=3)
    plt.plot(grad_firdif_3, label="$\omega = 3$")

    _, _, _, grad_firdif_5 = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                              early_stop_bool=early_stop, num_iters=num_iters,
                                              adapt_lr_bool=False,
                                              early_stop_threshold=early_stop_threshold,
                                              firdif_omega=5)
    plt.plot(grad_firdif_5, label="$\omega = 5$")

    _, _, _, grad_firdif_7 = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                              early_stop_bool=early_stop, num_iters=num_iters,
                                              adapt_lr_bool=False,
                                              early_stop_threshold=early_stop_threshold,
                                              firdif_omega=7)
    plt.plot(grad_firdif_7, label="$\omega = 7$")

    _, _, _, grad_firdif_9 = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                              early_stop_bool=early_stop, num_iters=num_iters,
                                              adapt_lr_bool=False,
                                              early_stop_threshold=early_stop_threshold,
                                              firdif_omega=9)
    plt.plot(grad_firdif_9, label="$\omega = 9$")

    _, _, _, grad_firdif_11 = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                               early_stop_bool=early_stop, num_iters=num_iters,
                                               adapt_lr_bool=False,
                                               early_stop_threshold=early_stop_threshold,
                                               firdif_omega=11)
    plt.plot(grad_firdif_11, label="$\omega = 11$")

    _, _, _, grad_firdif_13 = convar.convar_np(data, gamma, best_lambda, return_metric_gradients=True,
                                               early_stop_bool=early_stop, num_iters=num_iters,
                                               adapt_lr_bool=False,
                                               early_stop_threshold=early_stop_threshold,
                                               firdif_omega=13)
    plt.plot(grad_firdif_13, label="$\omega = 13$")

    plt.ylabel("$metric(gradient)$")
    plt.xlabel("iteration")
    # plt.title("Convar: metric(gradient) comparison")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    # plt.show()

    plt.savefig(r"F:\Uni Goethe\Informatik\BA\Latex\figure\gradients_firdif_omega.pgf")
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


