import torch
import numpy as np
import deconv_Dff
import sys
import h5py

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
        # deconv_Dff.deconv_torch_jit(example_data_path)
        # deconv_Dff.deconv(example_data_path)
        deconv_Dff.deconv_multicore(example_data_path)
        # deconv_Dff.deconv_multicore_ray(example_data_path)
        # np.show_config()

    if(mode == 5):
        with h5py.File(h5_path, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]

            d = f["ROI"]
            print(d[2])

            # Get the data
            # data = list(f[a_group_key])
            # print(data.keys())
