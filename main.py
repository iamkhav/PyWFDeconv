import torch
import numpy as np
import deconv_Dff
import deconv_testingAndPlots
import firdif
import convar
import sys
import h5py
from scipy.io import loadmat
from scipy.ndimage.filters import uniform_filter1d
import helpers
import wrappers
import helpers_pythran
import timeit
import datetime

example_data_path = r"Clancy_etal_fluorescence_example.mat"
h5_path = r"data_Jonas.hdf5"
npz_path = r"F0205_tseries_38_DF_by_F0_351_50_cut200_600_compressed.npz"

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

#Todo
# Warum sind die Values von beta0 nach 4. Nachkommastelle verschieden bei OPENBLAS scipy???

# To see full arrays
np.set_printoptions(threshold=sys.maxsize, precision=10)
torch.set_printoptions(threshold=sys.maxsize, precision=10)

# Set Torch default datatype to double, replicating the Matlab calculations
torch.set_default_dtype(torch.float64)

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
        """Tests and Benchmarks"""
        data_import = loadmat(example_data_path)
        cal_data = data_import["cal_data"]
        deconv_testingAndPlots.deconv_testing(cal_data=cal_data)

    if(mode == 3):
        """Plots"""
        data_import = loadmat(example_data_path)
        cal_data = data_import["cal_data"]
        deconv_testingAndPlots.deconv_for_plots(cal_data=cal_data)
        # deconv_testingAndPlots.deconv_T_and_P_plot(cal_data=cal_data)


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
        #Testing
        # testi = np.array([[1,2,3], [4, -2, 3.5], [-10, -12, 23.2]])
        # testi = np.array([2,3,4])
        # big_small_sum = helpers.biggest_pos_neg_sum(testi)

        # testo = np.random.rand(1000000).astype(np.float32)
        # t = timeit.Timer(lambda: helpers.moving_average(testo,3)).repeat(3, 100)
        # print(t)
        # print(testo.dtype)
        # t = timeit.Timer(lambda: helpers_pythran.moving_average(testo,3)).repeat(3, 100)
        # print(t)


        # Jonas hat ~ 3h gebraucht hierfür auf dem Minnesota Cluster
        npz_file = np.load(npz_path)

        # print(np.shape(npz_file))
        # print(npz_file.files)
        data = npz_file["data"]
        data = data[:200, :10000]
        print(f"Shape of Data: {np.shape(data)}")
        # firdif.firdif_np(data, 0.97, 3, printers=True)
        # convar.convar_np(data, 0.97, 1)
        convar.convar_np(data, 0.97, 1, earlyStop_bool=False)

    if(mode == 10):
        """Main Function using Wrappers"""

        # 1. Import data into a numpy ndarray, format it to have TxP dimension (P should be the pixels of the image per frame)
        npz_file = np.load(npz_path)
        data = npz_file["data"]
        data = data[:50, :10000]


        wrappers.find_best_lambda(data)
    # np.show_config()
