import numpy as np
# from numba import jit
from scipy.ndimage.filters import uniform_filter1d



def CleanDFFO(data, ROI=None):
    """
    This function is supposed to clean up input data by finding the proper start and ending, cutting off segments of continuos Inf or NaN before/after.
    1. We're checking for the first valid Pixel in the ROI bool mask. If none is passed, we'll just take a middle region pixel of input data.
    2. This pixel is then used to determine the first and last non invalid Pixel value.
    3. We're then discarding these invalid pixel value regions and returning data.
    Deyue said there would most likely be no invalid Pixel values in the middle of the series.
    -Amon
    :param data: 2d Matrix || Input Shape: (T, X, Y)
    :param ROI: 2d Bool Matrix that defines a Region of Interest, same shape as data || Input Shape: (X, Y)
    :return: Cleaned data
    """
    first_True_index = (0,0)
    if (ROI is not None):
        #Check if data and ROI have size mismatch
        if(np.shape(data)[-2:] != np.shape(ROI)[-2:]):
            raise Exception(f"Size mismatch: {np.shape(data)} and {np.shape(ROI)}")

        first_True_index = first_in_2dmat(ROI,True)

    first_True_x = first_True_index[0]
    first_True_y = first_True_index[1]

    start_or_end = "start"
    clean_start_index = 0
    clean_end_index = np.shape(data)[0]

    for ind, t in enumerate(data[:,first_True_x, first_True_y]):
        if(start_or_end == "start"):
            if(np.isfinite(t)):
                clean_start_index = ind
                start_or_end = "end"

        elif(start_or_end == "end"):
            if(not np.isfinite(t)):
                clean_end_index = ind
                break

    # print(data[clean_start_index:clean_end_index, first_True_x, first_True_y])
    return data[clean_start_index:clean_end_index, :, :]



#Todo look into Cython for first_in_2dmat?
# https://blog.paperspace.com/faster-numpy-array-processing-ndarray-cython/
#@jit(nopython=True)
def first_in_2dmat(input, searchvalue):
    """
    This is a helper function for cleanUpDfFo.
    :param input:
    :param searchvalue:
    :return:
    """
    for i in range(0,len(input)):
        for j in range(0,len(input)):
            if(input[i][j] == searchvalue):
                return (i,j)

    return None


def biggest_pos_neg_sum(in_array):
    """
    Find biggest positive and biggest |negative|.
    If no negative values, 0 is returned.

    Using this as a metric for Early Stop.

    :param in_array: Some numpy array.
    :return: Biggest Pos, Biggest |Neg|
    """
    curr_bigpos = 0
    curr_bigneg = 0
    for x in np.ravel(in_array):
        # if(x > 0 and x > curr_bigpos):
        #     curr_bigpos = x
        # if(x < 0 and x < curr_bigneg):
        #     curr_bigneg = x
        if(x > 0):
            if(x > curr_bigpos):
                curr_bigpos = x
        else:
            if(x < curr_bigneg):
                curr_bigneg = x

    return curr_bigpos + np.abs(curr_bigneg)

def moving_average(inny, span):
    """
    Should be an exact copy of the matlab smooth function when used with "moving" setting (which is default).
    :return: The smoother array
    """
    outy = uniform_filter1d(inny, span)
    outy[0] = inny[0]
    outy[-1] = inny[-1]
    return outy