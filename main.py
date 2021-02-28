import torch
import numpy as np
import deconv_Dff
import sys
import h5py
from scipy.io import loadmat
import helpers

example_data_path = r"Clancy_etal_fluorescence_example.mat"
h5_path = r"data_Jonas.hdf5"

"""
This is a straight (as straight as possible) translation from the matlab code included in the following paper:
Todo


"""

#Todo
# Maybe everything else should also have a Pytorch version.. ?? Numpy CPU is very slow, slower than matlab on some Calcs
# Warum sind die Values von beta0 nach 4. Nachkommastelle verschieden bei OPENBLAS scipy???

# To see full arrays
np.set_printoptions(threshold=sys.maxsize, precision=10)
torch.set_printoptions(threshold=sys.maxsize, precision=10)

# Set Torch default datatype to double, replicating the Matlab calculations
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    mode = 1
    if(mode == 1):
        # Load data
        # cal_data should be formatted in a way that axis=0 is T while axis=1 is x,y flattened of the input image
        data_import = loadmat(example_data_path)
        cal_data = data_import["cal_data"]

        # deconv_Dff.deconv_testing(cal_data=cal_data)

        deconv_Dff.deconv(cal_data=cal_data)
        # deconv_Dff.deconv_multicore(cal_data=cal_data)


        # deconv_Dff_experimental.deconv_multicore_ray(cal_data=cal_data)
        # deconv_Dff_experimental.deconv_torch_jit(cal_data=cal_data)

        # np.show_config()

    if(mode == 5):
        # Main function example
        #

        # Read files
        with h5py.File(h5_path, "r") as f:
            # List all groups
            print(f"Keys: {f.keys()}")

            # Get data and ROI
            df_fo = np.array(f["DF_by_FO"])
            ROI = np.array(f["ROI"])

        # Clean the data (if you suspect or know about Inf/NaN regions)
        cleaned_df_fo = helpers.CleanDFFO(df_fo, ROI=ROI)
