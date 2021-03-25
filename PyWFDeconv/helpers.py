import numpy as np
# from numba import jit
from scipy.ndimage.filters import uniform_filter1d
import warnings
import multiprocessing


def CleanDFFO(data, ROI=None):
    """
        This function is supposed to clean up input data by finding the proper start and ending, cutting off segments of continuos Inf or NaN before/after.
        1. We're checking for the first valid Pixel in the ROI bool mask. If none is passed, we'll just take a middle region pixel of input data.
        2. This pixel is then used to determine the first and last non invalid Pixel value.
        3. We're then discarding these invalid pixel value regions and returning data.

        Deyue said there would most likely be no invalid Pixel values in the middle of the series.

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




def moving_average(inny, span):
    """
        Should be an exact copy of the matlab smooth function when used with "moving" setting (which is default).
        USED IN FIRDIF

    :param inny: Input array (1d)
    :param span: Span, works with all positive integers but should be an odd number
    :return: The smoothed array
    """
    if(span % 2 == 0):
        warnings.warn("Span should be an odd number >0")
    outy = uniform_filter1d(inny, span)
    outy[0] = inny[0]
    outy[-1] = inny[-1]
    return outy


def scale_to(x, max_desired, min_desired, max_x=1, min_x=0):
    """
        Scales an x to a desired range.
        You have to pass the maximum and minimum possible value for x.

        USED FOR ADAPTIVE LR

    :param x:
    :param max_x:
    :param min_x:
    :param max_desired:
    :param min_desired:
    :return:
    """
    return ((x - min_x) / (max_x - min_x)) * (max_desired - min_desired) + min_desired


def normalize_1_minus1(arrayo):
    """
        Wraps the scale_to function and scales from 1 to -1

    :param arrayo:
    :return:
    """
    maxo = np.max(arrayo)
    mino = np.min(arrayo)
    outo = [scale_to(x, 1, -1, maxo, mino) for x in arrayo]

    return outo

def determine_num_workers():
    """
        Automatic determination of cores to be used for multiple process spawns of convar (just a recommendation).

    :return: Integer with workers to use
    """
    num_cores = multiprocessing.cpu_count()
    print(f"Number of cores: {num_cores}")

    if(num_cores > 8):
        print("Number of cores above 8, picking num_worker = 8.")
        print("If CPU load isn't 100% consistently, try increasing num_workers.")
        return 8
    else:
        print(f"Using all available cores: {num_cores}")
        print(f"Try {num_cores - 1} if you encounter problems or want to use the Computer while running this.")
        return num_cores