import numpy as np
import PyWFDeconv as wfd
from scipy.io import loadmat
import sys
import h5py
import datetime

#Merav Data
merav_data_path = r"ExampleData/Clancy_etal_fluorescence_example.mat"
#Jonas Data
jonas_data_path = r"ExampleData/F0205_tseries_38_DF_by_F0_351_50_cut200_600_compressed.npz"


# 1. Import data into a numpy ndarray, expected format: 2d TxP ndarray (T:frames, P:pixels)
npz_file = np.load(jonas_data_path)
data = npz_file["data"]
data = data[:, :]


if __name__ == '__main__':
    print(f"Launching program.. at {datetime.datetime.now()}")
    example = 2

    if(example==1):
        # 1. Import data into a numpy ndarray, expected format: 2d TxP ndarray (T:frames, P:pixels)
        npz_file = np.load(jonas_data_path)
        data = npz_file["data"]

        # 2. Determine which lambda yields best results
        # Jonas data, Gamma adjusted
        best_lambda = wfd.find_best_lambda(data[:200, :500], gamma=0.92, convar_num_iters=2000, convar_mode="adapt")

        # 3. Deconvolve using best lambda
        wfd.deconvolve(data, gamma=0.02, best_lambda=best_lambda, convar_mode="adapt")

    if(example==2):
        # 1. Import data into a numpy ndarray, expected format: 2d TxP ndarray (T:frames, P:pixels)
        data_import = loadmat(merav_data_path)
        data = data_import["cal_data"]

        # 2. Determine which lambda yields best results
        best_lambda = wfd.find_best_lambda(data, convar_num_iters=2000, convar_mode="adapt")

        # 3. Deconvolve using best lambda
        wfd.deconvolve(data, best_lambda=best_lambda, convar_mode="adapt")