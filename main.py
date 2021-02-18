import torch
import numpy as np
import deconv_Dff
import sys

example_data_path = r"Clancy_etal_fluorescence_example.mat"

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
    deconv_Dff.deconv(example_data_path)
    # np.show_config()
