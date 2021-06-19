import PyWFDeconv.deconv_Dff as deconv_Dff
import PyWFDeconv.plot_deconv_testingAndPlots as deconv_testingAndPlots
import PyWFDeconv.firdif as firdif
import PyWFDeconv.convar as convar
import PyWFDeconv.helpers as helpers
# import PyWFDeconv.convar_deprecated as convar_deprecated
import PyWFDeconv.early_stops as early_stops

# Plots
import PyWFDeconv.plot_lineperline_convar as plot_lineperline_convar
import PyWFDeconv.plot_matmul_complexity as plot_matmul_complexity
import PyWFDeconv.plot_slicedT_vs_regular as plot_slicedT_vs_regular
import PyWFDeconv.plot_normalized_vs_regular as plot_normalized_vs_regular
import PyWFDeconv.plot_generic_functions as plot_generic_functions
import PyWFDeconv.plot_code_excerpts as plot_code_excerpts
import PyWFDeconv.plot_gradient_improvements as plot_gradient_improvements

import PyWFDeconv as wfd

# import torch
import numpy as np
from scipy.io import loadmat
import sys
import h5py
import timeit
import datetime
# import cProfile
# import time

example_data_path = r"ExampleData/Clancy_etal_fluorescence_example.mat"
h5_path = r"ExampleData/data_Jonas.hdf5"
npz_path = r"ExampleData/F0205_tseries_38_DF_by_F0_351_50_cut200_600_compressed.npz"

"""
This is a translation and enhancement of the matlab code included in the following paper:
XXX


Notes:
~~~ Comments
In the originally translated methods I included all (or most) of the original comments.
Whenever I (Amon) would add a comment, I'd add a "-Amon" flag to the comment.
In the new or "enhanced" parts of the code (e.g. wrappers.py, helpers.py) I just commented without the flag.


~~~ Test
Blabla


~~~ Cores
It makes a lot of sense to specify the number of processes to be spawned (num_workers argument).
Matrix multiplications uses multiple cores implicitly (if the library doesn't use very inefficient routines, more on that later).
You should monitor CPU load % while using a single worker to see how a single core keeps up with the matrix multiplication instructions.

"""

# To see full arrays
# np.set_printoptions(threshold=sys.maxsize, precision=10)
# torch.set_printoptions(threshold=sys.maxsize, precision=10)

# Set Torch default datatype to double, replicating the Matlab calculations
# torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    mode = 10
    print(f"Launching program.. at {datetime.datetime.now()}")

    if(mode == 1):
        """Current Main"""
        # Load data
        # cal_data should be formatted in a way that axis=0 is T while axis=1 is x,y flattened of the input image
        data_import = loadmat(example_data_path)
        cal_data = data_import["cal_data"]

        deconv_Dff.deconv(cal_data=cal_data)
        # deconv_Dff.deconv_multicore(cal_data=cal_data)
        # deconv_Dff_experimental.deconv_multicore_ray(cal_data=cal_data)
        # deconv_Dff_experimental.deconv_torch_jit(cal_data=cal_data)


    if(mode == 2):
        """Tests/Benchmarks"""
        data_import = loadmat(example_data_path)
        cal_data = data_import["cal_data"]
        deconv_testingAndPlots.deconv_testing(cal_data=cal_data)


    if(mode == 3):
        """Plots"""
        data_import = loadmat(example_data_path)
        cal_data = data_import["cal_data"]
        # deconv_testingAndPlots.deconv_for_plots(cal_data=cal_data)
        # deconv_testingAndPlots.deconv_T_and_P_plot(cal_data=cal_data)

        plot_code_excerpts.scale_t_scale_p()



    if(mode == 5):
        """Main function example"""
        # Read files
        with h5py.File(h5_path, "r") as f:
            # List all groups
            print(f"Keys: {f.keys()}")

            # Get data and ROI
            df_fo = np.array(f["DF_by_FO"])
            ROI = np.array(f["ROI"])

        # Clean the data (if you suspect or know about Inf/NaN regions)
        cleaned_df_fo = helpers.CleanDFFO(df_fo, ROI=ROI)


    if(mode == 6):
        """Testing"""
        # testi = np.array([[1,2,3], [4, -2, 3.5], [-10, -12, 23.2]])
        # testi = np.array([2,3,4])
        # big_small_sum = early_stops.biggest_pos_neg_sum(testi)
        # t = timeit.Timer(early_stops.biggest_pos_neg_sum(testi)).repeat(3, 100)
        # print(t)


        data_import = loadmat(example_data_path)
        cal_data = data_import["cal_data"] #* 100
        # print(np.shape(cal_data))
        # cal_data = data_import["cal_data"]

        #Todo:  Daten sammeln, um zu zeigen, dass an kleineren P die T splitting wie vermutet ist
        cal_data = cal_data[:, :]


        gamma = 0.97
        # cal_data = cal_data[::2]
        # gamma = 1 - (1 - gamma) / 0.5
        # plot_code_excerpts.convar_cow(cal_data, gamma, 1)
        # convar.convar_np(cal_data, gamma, 1, early_stop_bool=False)
        # convar.convar_np(cal_data, gamma, 1)
        # cProfile.run("convar.convar_np(cal_data, gamma, 1)")
        # convar.convar_np(cal_data, gamma, 1, adapt_lr_bool=True)
        # convar.convar_np(cal_data, gamma, 1, adapt_lr_bool=True, early_stop_bool=False)


        # Jonas hat ~ 3h gebraucht hierfür auf dem Minnesota Cluster
        npz_file = np.load(npz_path)

        gamma = 0.92
        data = npz_file["data"]
        data = data[:400, :]
        # data = data[:100, :10000]
        # data = data[:50, :5000]

        firdif, _, _ = firdif.firdif_np(data, 0.92, 3)
        # a,_,_ = wfd.deconvolve(data, best_lambda=1, gamma=0.92, adapt_lr_bool=True, convar_earlystop_threshold=0.000000001, convar_earlystop_metric=early_stops.mean_square, num_workers=0)
        a,_,_ = wfd.deconvolve(data, best_lambda=1, gamma=0.92, adapt_lr_bool=True, convar_earlystop_threshold=0.0001, num_workers=0)
        b,_,_ = wfd.deconvolve(data, best_lambda=1, gamma=0.92, adapt_lr_bool=True, convar_earlystop_threshold=0.00001, num_workers=0)
        plot_generic_functions.plot_r_finals((data,a,b))


        # start = time.time()
        # for x in range(0,10):
        #     # convar.convar_np(data, gamma, 5, adapt_lr_bool=True)
        #     wfd.deconvolve(data, gamma, 5, adapt_lr_bool=True, num_workers=0)
        # print(time.time() - start)

        # convar.convar_np(data, gamma, 1, num_iters=10000, early_stop_bool=False)
        # convar.convar_half_torch(data, gamma, 1)

        # convar.convar_np(data, gamma, 1, early_stop_bool=False)

        # plot_code_excerpts.convar_cow(data, gamma, 1)

    if(mode == 7):
        """Plot functions"""

        # Data import
        # Merav 400x50
        data_import = loadmat(example_data_path)
        cal_data = data_import["cal_data"]

        # Jonas 400x10334
        npz_file = np.load(npz_path)
        data = npz_file["data"]
        data = data[:, :]

        ## Matmul runtimes with scaled T
        # plot_matmul_complexity.plot_or_bench_matmul_runtime()

        ## Convar runtimes with scaled T
        # plot_matmul_complexity.plot_or_bench_convar_runtimes()

        ## Convar normalized input data vs regular
        # plot_normalized_vs_regular.compare_normalized_vs_regular(cal_data)

        # Sliced T vs regular
        # plot_slicedT_vs_regular.compare_slice_vs_regular(data)

        # Print dirac * calcium
        # plot_code_excerpts.dirac_calcium_conv()

        # Line per Line timings
        # plot_lineperline_convar.convar_np_at(data, 0.97, 1)

        # Show Gradient with sequentially added features
        # plot_gradient_improvements.show_plots(1)
        # plot_code_excerpts.meanGradient_over_t(cal_data, 0.97, 1) # OLD WAY OF DOING IT

        # Show divergence
        # plot_gradient_improvements.divergence_example(0)

        # Test fuer Matthias Kaschube
        plot_generic_functions.plot_every_trace_every_way_and_save(helpers.normalize_1_0(data))

    if(mode == 10):
        """Main Function using Wrappers"""

        # 1. Import data into a numpy ndarray, expected format: 2d TxP ndarray (T:frames, P:pixels)
        npz_file = np.load(npz_path)
        data = npz_file["data"]
        data = data[:, :]
        # data_import = loadmat(example_data_path)
        # cal_data = data_import["cal_data"]
        # data = cal_data


        # 2. Determine which lambda yields best results
        # lambda_list = wfd.generate_lambda_list(0.5,4,0.1)
        # best_lambda = wfd.find_best_lambda(data[:200, :1000], gamma=0.92, convar_num_iters=2000, adapt_lr_bool=True, num_workers=0)        # Jonas data, Gamma adjusted
        # best_lambda = wfd.find_best_lambda(data[:200, :1000], gamma=0.92, convar_num_iters=2000, adapt_lr_bool=True, binary_seach_find=True, all_lambda=lambda_list)

        # 3. Deconvolve using best lambda
        deconvolved, _, _ = wfd.deconvolve(data[:, :], gamma=0.92, best_lambda=5, adapt_lr_bool=True, num_workers=8, convar_earlystop_threshold=0.000001, printers=2)

        # _,_,_ = firdif.firdif_np(data, gamma=0.92, smt=3)


    # np.show_config()

