import numpy as np
import PyWFDeconv as wfd
from scipy.io import loadmat
import datetime

#Merav Data
merav_data_path = r"ExampleData/Clancy_etal_fluorescence_example.mat"
#Jonas Data
jonas_data_path = r"ExampleData/F0205_tseries_38_DF_by_F0_351_50_cut200_600_compressed.npz"


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
        best_lambda = wfd.find_best_lambda(data[:200, :1000], gamma=0.92, convar_num_iters=2000, adapt_lr_bool=True)

        # 3. Deconvolve using best lambda
        deconvolved, _, _ = wfd.deconvolve(data[:, :], gamma=0.92, best_lambda=best_lambda, adapt_lr_bool=True, num_workers=None, convar_earlystop_threshold=0.0000001)

    if(example==2):
        # 1. Import data into a numpy ndarray, expected format: 2d TxP ndarray (T:frames, P:pixels)
        data_import = loadmat(merav_data_path)
        data = data_import["cal_data"]

        # 2. Determine which lambda yields best results
        best_lambda = wfd.find_best_lambda(data[:, :], gamma=0.97, convar_num_iters=2000, adapt_lr_bool=True)

        # 3. Deconvolve using best lambda
        deconvolved, _, _ = wfd.deconvolve(data[:, :], gamma=0.97, best_lambda=best_lambda, adapt_lr_bool=True)
