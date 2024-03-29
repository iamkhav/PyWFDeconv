# from torch.multiprocessing import Pool
import multiprocessing
from multiprocessing import Pool, cpu_count
import numpy as np
from functools import partial
import time
from . import (
    helpers,
    convar,
    firdif
)
from math import ceil, floor
from matplotlib import pyplot as plt

# Original lambdas if the user wants to call and modify them
original_all_lambda = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2, 3, 5, 7, 10, 20, 40, 80]


"""
This contains wrappers that automate most of the work you'd have to do manually.
Using Python function-defaults you'll be able to configure your optimal usage.

Important: This is what gets included when you include this package.

Originally Merav Stern et al. structured the main procedure into a matlab file called "deconv_Dff_data_all_steps_using_continuously_varying.m" which was meant to be a template.
My wrappers are supposed to be a substitute aimed at our lab's requirements.
"""

"""
Argument List: 
data                - input numpy ndarray TxP
    
gamma               - calcium decay rate, default: 0.97 (single neuron, based on 40Hz measurments in Gcamp6f mice)
                    
num_workers         - Number of parallel processes to be used
                    - If num_workers is left at None, helpers.determine_num_workers() will recommend a num_workers and use it.
    
all_lambda          - Lambdas to be tested

adapt_lr_bool       - True|False
                    - False  = One conventional run with early Stop activated, First Difference method for output matrix init and standard LR
                    - True     = Using Adaptive LR
                    
convar_num_iters    - number of iterations for convar to run 

early_stop_bool     - Sets Early Stop function for convar

printers            - Settings printers to false reduces prints to bare minimum
                    - 0 | 1 | 2
                    - meaning => "silent"|"minimize"|"full"
                    
binary_seach_find   - Uses binary search style algorithm for lambda search.. 
                    - all_lambda needs to be sorted!!!
                    - all_lambda needs to have even spacing!!! (best to initialize with wfd.generate_lambda_list
                    - Can be useful when lambdas are granular and Convars take a long time
                    - Cuts down number of Convars called from n to logn
                    - Uses a Cache system that further reduces number of Convars called
"""


def __do_work_best_lambda(partial_convar_f, data, gamma, all_lambda, num_workers=1, printers=1):
    """Docstring"""
    # Error Checking
    if(num_workers <= 0):
        num_workers = 1
    odd_traces = data[0::2]
    even_traces = data[1::2]

    # number of points in each odd/even calcium trace
    T = np.shape(odd_traces)[0]
    rep = np.shape(odd_traces)[1]

    # will be used later to reconstruct the calcium from the deconvoled rates
    Dinv = np.zeros((T, T))

    for k in range(0, T):
        for j in range(0, k + 1):
            exp = (k - j)
            Dinv[k][j] = gamma ** exp

    # saving the results
    # here the penalty (l2) is the same as the fluctuations (l2)
    penalty_size_convar = np.zeros((len(all_lambda), rep))
    calcium_dif_convar = np.zeros((len(all_lambda), rep))

    start = time.time()
    with Pool(num_workers) as p:
        results = p.map(partial_convar_f, all_lambda)
    end = time.time()
    # if(printers > 0): print("All Convars time:", round(end - start, 2))

    c = 0
    for k in results:
        # Unpacking variables from multiprocessing
        r, r1, beta0 = k[0], k[1], k[2]
        # calculating the changes in spiking rate in each deconvolve trace
        r_diff = np.diff(r[1:], axis=0)
        # calculating the penalty in each trace
        penalty_size_convar[c] = np.mean(np.square(r_diff), axis=0)
        # reconstruct the calcium
        c_odd = np.matmul(Dinv, np.vstack((r1, r)))
        calcium_dif_convar[c] = np.mean(np.abs(c_odd+beta0-even_traces), axis=0)
        c += 1

    temp = np.mean(calcium_dif_convar, axis=1)
    min_error_convar = np.min(temp)
    best_lambda_convar_indx = np.argmin(temp)
    best_lambda_convar = all_lambda[best_lambda_convar_indx]
    if(False):   # Plot Lambda curve
        print(temp)
        for x in range(0,len(temp)):
            print(temp[x])
            print(round(all_lambda[x],2))
            print("---------")
        plt.rcParams.update({'font.size': 12})
        plt.rcParams["figure.figsize"] = (6.5 * 0.7, 6.5 * 0.75 * 0.7)  # Bachelor Thesis page width and 4.32=good looking
        # plt.rcParams.update({'font.size': 11})
        # plt.rcParams["figure.figsize"] = (6.5 * 0.6, 6.5 * 0.75 * 0.6)  # Bachelor Thesis page width and 4.32=good looking
        # for x in range(0,len(lambdalist)):
        #     lambdalist[x] = lambdalist[x]/1.25
        plt.plot(all_lambda, temp)
        # plt.yscale("log")
        plt.ylabel("Score")
        plt.xlabel("$\lambda$")
        plt.tight_layout()
        # plt.savefig(r"F:\Uni Goethe\Informatik\BA\Latex\figure\lambdasearch_scores.pgf")  # , bbox_inches='tight')
        # plt.savefig(r"F:\Uni Goethe\Informatik\BA\Latex\figure\lambdasearch_scores_small.pgf")  # , bbox_inches='tight')
        plt.show()
    return best_lambda_convar, min_error_convar, best_lambda_convar_indx

def generate_lambda_list(start_range, end_range, increment):
    """
    General helper function to help generate a lambda list for find_best_lambda.
    Included here (instead of helpers.py) because it might be frequently used by users.
    Of course wrapping a numpy function doesn't really justify a separate function but hopefully this saves people some time.
    :param start_range:
    :param end_range:
    :param increment:
    :return:
    """
    return list(np.arange(start_range, end_range, increment))

def find_best_lambda(data, gamma=0.97, num_workers=None, all_lambda=None, times_100=False, normalize=True,
                     adapt_lr_bool=True, convar_algo="numpy", convar_num_iters=2000, early_stop_bool=False, firdif_best_omega=None,
                     binary_seach_find_bool=False,
                     printers=1):
    """Docstring"""

    if(printers>0):
        print("------------------------------------------------------")
        print(f"{'FINDING BEST LAMBDA':^60}")
        print(f"{'Shape of input data:':^40} {np.shape(data)}")
        if(adapt_lr_bool):  print(f"{'Using adaptive LR':^40}")
        else:               print(f"{'Using regular LR':^40}")
        if(binary_seach_find_bool):
            print(f"{'Using Binary Search Method':^40}")
            print(f"{'(Make sure you have even spacing in Lambda List..)':^40}")


    min_error_convar = 0
    best_lambda_convar = 0
    workers_printers = False
    # Workers Init
    if(num_workers == None):
        if(printers == 2): workers_printers = True
        num_workers = helpers.determine_num_workers(printers=workers_printers)
        if(printers > 0): print(f"Argument num_workers left blank, auto-determined {num_workers}..")
    elif(num_workers == -1):
        num_workers = multiprocessing.cpu_count()-1


    # Normalize Data to [0,1]
    if(normalize): data = helpers.normalize_1_0(data)


    # Carry over from MatLab
    if(times_100):data = data * 100

    # Cut Lambdas, divisible by 2 for MP
    # Not defining this in def: line because mutable arguments are bad style
    if(all_lambda==None): all_lambda = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2, 3, 5, 10, 20]

    # Need to fit for half frequency because we're taking half the data
    ratio = 0.5
    gamma = 1 - (1 - gamma) / ratio
    odd_traces = data[0::2]

    # Create partial for MP
    partial_f = partial(convar.convar_np, odd_traces, gamma, num_iters=convar_num_iters, early_stop_bool=early_stop_bool, adapt_lr_bool=adapt_lr_bool, printers=workers_printers, firdif_omega=firdif_best_omega)

    start = time.time()
    if(not binary_seach_find_bool):
        # Go through all lambdas
        best_lambda_convar, min_error_convar, _ = __do_work_best_lambda(partial_f, data, gamma, all_lambda, num_workers, printers)

    if (binary_seach_find_bool):
        #Todo check for non-evenly spaced input_lambda_list

        # Binary Search Style Lambda Search
        left = 0
        right = len(all_lambda) - 1
        num_workers = 1
        # Initialize Cache
        cache = [-1] * len(all_lambda)
        temp_best_index = -1
        number_of_convar_calls = 0
        which_side_better = ""

        while(right-left > 0):
            # As long as we have elements to look at..
            mid = ceil((left + right) / 2)
            left_mid = floor((left + mid) / 2)
            right_mid = ceil((right + mid) / 2)
            print(left, mid, right)
            print(f"Convar on {left_mid} {right_mid}")

            # Cache check
            if(cache[left_mid] == -1 and cache[right_mid] == -1):
                best_lambda_convar, min_error_convar, temp_best_index = __do_work_best_lambda(partial_f, data, gamma, [all_lambda[left_mid],all_lambda[right_mid]], num_workers, printers == 0)
                number_of_convar_calls += 2
                if(temp_best_index == 0):
                    print("Left side better.")
                    which_side_better = "l"
                    # Write to cache
                    cache[left_mid] = min_error_convar
                if(temp_best_index == 1):
                    print("Right side better.")
                    which_side_better = "r"
                    # Write to cache
                    cache[right_mid] = min_error_convar

            elif(cache[left_mid] != -1 and cache[right_mid] != -1):
                print("Left AND Right were cached")
                if(cache[left_mid] < cache[right_mid]):
                    print("Left side better.")
                    which_side_better = "l"
                    best_lambda_convar = all_lambda[left_mid]
                else:
                    print("Right side better.")
                    which_side_better = "r"
                    best_lambda_convar = all_lambda[right_mid]

            elif(cache[right_mid] != -1):
                print("Right was cached")
                left_best_convar, left_min_error_convar, _ = __do_work_best_lambda(partial_f, data, gamma, [all_lambda[left_mid]], num_workers, printers == 0)
                number_of_convar_calls += 1
                if(left_min_error_convar < cache[right_mid]):
                    print("Left side better.")
                    which_side_better = "l"
                    cache[left_mid] = left_min_error_convar
                    best_lambda_convar = left_best_convar
                else:
                    print("Right side better.")
                    which_side_better = "r"
                    best_lambda_convar = all_lambda[right_mid]

            elif (cache[left_mid] != -1):
                print("Left was cached")
                right_best_convar, right_min_error_convar, _ = __do_work_best_lambda(partial_f, data, gamma, [all_lambda[right_mid]], num_workers, printers == 0)
                number_of_convar_calls += 1
                if(cache[left_mid] > right_min_error_convar):
                    print("Right side better.")
                    cache[right_mid] = right_min_error_convar
                    best_lambda_convar = right_best_convar
                    which_side_better = "r"

                else:
                    print("Left side better.")
                    best_lambda_convar = all_lambda[left_mid]
                    which_side_better = "l"

            # Break condition before setting new boundary
            if (right - left == 1):
                break
            # Set the new mid
            if(which_side_better == "l"):
                right = mid
            elif(which_side_better == "r"):
                left = mid

            print("..")
        print(f"Number of Convar Calls: {number_of_convar_calls}")

    if(printers > 0):
        print("------------------------------------------------------")
        print("------------------------------------------------------")
        print(f"{'Time taken:':^40} {round(time.time() - start, 2)}s")
        # print(f"{'Min error Convar:':^40} min_error_convar")
        print(f"{'Best Lambda:':^40} {best_lambda_convar}")

    return best_lambda_convar

def find_best_omega(data, gamma=0.97, normalize=True):
    """
    Wrapper for firdif.firdif_best_window to find best smoothing factor omega (or smt).
    :param data:
    :param gamma:
    :param normalize:
    :return:
    """
    # Normalize Data to [0,1]
    if(normalize): data = helpers.normalize_1_0(data)
    return firdif.firdif_best_window(data, gamma)

"""
Argument List:
data                - input numpy ndarray TxP
    
gamma               - calcium decay rate (single neuron, based on 40Hz measurments in Gcamp6f mice)
    
best_lambda         - determined in find_best_lambda, default: 1
    
num_workers         - Number of parallel processes to be used
                    - If num_workers is left at None, helpers.determine_num_workers() will recommend a num_workers and use it. 
            
                    
adapt_lr_bool       - True|False
                    - False  = One conventional run with early Stop activated, First Difference method for output matrix init and standard LR
                    - True     = Using Adaptive LR
                    
convar_num_iters    - number of iterations for convar to run 

Note: Using overlap blend and steal together only makes sense if the blend area is bigger than the steal area since else the blended frames get "stolen". Using one of the two is the better choice (blend imo).

"""

def deconvolve(
    data, gamma=0.97, best_lambda=1, times_100=False, normalize=True,
    num_workers=None,
    chunk_t_bool=False, chunk_size=100, chunk_overlap=10, chunk_overlap_steal = 0, chunk_overlap_blend = 0,
    adapt_lr_bool=True, convar_num_iters=10000, convar_earlystop_metric=None, convar_earlystop_threshold=None, firdif_best_omega=None,
    printers=1):
    """Docstring"""


    # Normalize Data to [0,1]
    if(normalize): data = helpers.normalize_1_0(data)

    # Init returns
    r_final,r1,beta_0 = np.empty(0),0,0

    workers_printers = False
    # Workers Init
    if(num_workers == None):
        if(printers == 2):
            workers_printers = True
        num_workers = helpers.determine_num_workers(printers=workers_printers)
        if(printers > 0): print(f"Argument num_workers left blank, auto-determined {num_workers}..")
    elif(num_workers == -1):
        num_workers = multiprocessing.cpu_count()-1

    # Carry over from MatLab
    if(times_100):data = data * 100

    # Print Info about current run
    if(printers>0):
        print("------------------------------------------------------")
        print(f"{'DECONVOLUTION':^60}")
        print(f"{'Shape of input data:':^40} {np.shape(data)}")
        if(adapt_lr_bool):  print(f"{'Using adaptive LR':^40}")
        else:               print(f"{'Using regular LR':^40}")

        if(chunk_t_bool and num_workers>0):
            # Chunk T, MP on T
            print(f"{'Chunking T and multiprocessing on T..':^40}")
            # print(f"{'Number of Workers:':^40} {num_workers}")
        elif (chunk_t_bool):
            # Chunk T, no MP
            print(f"{'Chunking T and no multiprocessing':^40}")
        elif (not chunk_t_bool and num_workers > 0):
            # Chunk P, MP on P
            print(f"{'Chunking P and multiprocessing on P..':^40}")
            # print(f"{'Number of Workers:':^40} {num_workers}")
        else:
            print(f"{'No Chunks, no multiprocessing.':^40}")


    #-----------------------------------
    # STARTING DECONV

    # Creating convar function with prefilled arguments (so Pool.map can be used)
    lambda_convar = partial(__convar_arg_reorganizer, gamma, best_lambda, convar_num_iters, adapt_lr_bool, convar_earlystop_metric, convar_earlystop_threshold, workers_printers, firdif_best_omega)


    start = time.time()
    if(chunk_t_bool and num_workers>0):
        # Chunk T, MP on T
        num_loops = ceil(np.shape(data)[0] / chunk_size)
        data_list = []
        # Create list of data for MP
        for i in range(0, num_loops):
            if (i == 0):
                data_list.append(data[(i * chunk_size): (i + 1) * chunk_size, :])
                continue
            data_list.append(data[(i * chunk_size) - chunk_overlap: (i + 1) * chunk_size, :])
        # MP
        with Pool(num_workers) as p:
            results = np.array(p.map(lambda_convar, data_list), dtype=object)

        # Unpack and stitch results together
        c = 0
        for x in results:
            if(c == 0):
                r_final = x[0]
                c += 1
                continue

            # Overlap blend
            if(chunk_overlap_blend > 0):
                r_final[-chunk_overlap_blend:] = (r_final[-chunk_overlap_blend:] + x[0][chunk_overlap - (1 + chunk_overlap_blend): chunk_overlap - 1]) / 2

            if(chunk_overlap_steal > 0):
                # Overlap steal
                r_final = np.concatenate((r_final[:-chunk_overlap_steal], x[0][chunk_overlap - (1+chunk_overlap_steal):]), axis=0)
            else:
                # Regular chunk concatenation
                r_final = np.concatenate((r_final, x[0][chunk_overlap - 1:]), axis=0)
            c += 1

    elif(chunk_t_bool):
        # Chunk T, no MP
        num_loops = ceil(np.shape(data)[0] / chunk_size)
        for i in range(0, num_loops):
            if (i == 0):
                r_final, _, _ = lambda_convar(data[(i * chunk_size): (i + 1) * chunk_size, :])
                continue
            temp_r, _, _ = lambda_convar(data[(i * chunk_size) - chunk_overlap: (i + 1) * chunk_size, :])

            # Overlap blend
            if(chunk_overlap_blend > 0):
                r_final[-chunk_overlap_blend:] = (r_final[-chunk_overlap_blend:] + temp_r[chunk_overlap - (1 + chunk_overlap_blend): chunk_overlap - 1]) / 2

            if(chunk_overlap_steal > 0):
                # Overlap steal
                r_final = np.concatenate((r_final[:-chunk_overlap_steal], temp_r[chunk_overlap - (1+chunk_overlap_steal):]))
            else:
                # Regular chunk concatenation
                r_final = np.concatenate((r_final, temp_r[chunk_overlap - 1:]))

    elif(not chunk_t_bool and num_workers>0):
        # Chunk P, MP on P
        data_list = helpers.chunk_list_axis1(data, num_chunks=num_workers)
        # MP
        with Pool(num_workers) as p:
            # Have to cast manually and pick dtype=object because numpy deprecated (or plans to deprecate) uneven chunks in np.arrays
            results = np.array(p.map(lambda_convar, data_list), dtype=object)

        # Unpack and stitch results together
        c = 0
        for x in results:
            if(c == 0):
                r_final = x[0]
                c += 1
                continue
            r_final = np.concatenate((r_final, x[0]), axis=1)
            c += 1

    else:
        # No Chunks, no MP
        r_final,r1,beta_0 = lambda_convar(data)

    end = time.time()

    if(printers > 0):
        print("------------------------------------------------------")
        print("------------------------------------------------------")
        print(f"{'Time taken:':^40} {round(end - start, 2)}s")
        print("------------------------------------------------------")

    return r_final,r1,beta_0

def __convar_arg_reorganizer(gamma, _lambda, num_iters, adapt_lr_bool, early_stop_metric_f, early_stop_threshold, printers, firdif_best_omega, data):
    """Needed for functool.partials (maybe?).. Else I'll use lambdas.."""
    return convar.convar_np(data, gamma, _lambda, num_iters=num_iters, adapt_lr_bool=adapt_lr_bool, early_stop_metric_f=early_stop_metric_f, early_stop_threshold=early_stop_threshold, printers=printers, firdif_omega=firdif_best_omega)

def fit_gamma(new_frame_rate):
    #gamma               - calcium decay rate (single neuron, based on 40Hz measurments in Gcamp6f mice)
    gamma = 0.97
    ratio = new_frame_rate / 40
    return 1 - (1 - gamma) / ratio
