from import_library import *
example_data_path = r"F:\Uni Goethe\Informatik\BA\Python_Wide_Field_Deconvolution\Clancy_etal_fluorescence_example.mat"

"""
This is a straight (as straight as possible) translation from the matlab code included in the following paper:
Todo


"""

#Todo
# Implement convar alternative in Pytorch (for cuda acceleration)
# Maybe everything else should also have a Pytorch version.. ?? Numpy CPU is very slow, slower than matlab on some Calcs
# Warum sind die Values von beta0 nach 4. Nachkommastelle verschieden je nach Implementation?!?!?!

# To see full arrays
np.set_printoptions(threshold=sys.maxsize, precision=10)
torch.set_printoptions(threshold=sys.maxsize, precision=10)


if __name__ == '__main__':
    deconv_Dff.deconv(example_data_path)
    # np.show_config()