from torch.multiprocessing import Pool
# from multiprocessing import Pool
import numpy as np
from functools import partial
import time
from . import (
    helpers,
    convar
)

# Original lambdas if the user wants to call and modify them
original_all_lambda = [80, 40, 20, 10, 7, 5, 3, 2, 1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]


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
    
convar_mode         - "standard"|"cow"
                    - standard  = One conventional run with early Stop activated, First Difference method for output matrix init and standard LR
                    - cow       = Two runs, first adds the experimental adaptive LR, feeding the output of that into a standard convar run (hopefully instantly terminating)
                    
convar_algo         - "numpy"|"scipyBLAS"|"torchCUDA"

convar_num_iters    - number of iterations for convar to run 

early_stop_bool     - Sets Early Stop function for convar

printers            - Settings printers to false reduces prints to bare minimum
"""



def find_best_lambda(data, gamma=0.97, num_workers=None, all_lambda=None,
                     convar_mode="standard", convar_algo="numpy", convar_num_iters=2000, early_stop_bool=False,
                     printers=True):
    """Docstring"""

    if(printers): print("------------------------------------------------------")

    # Workers Init
    if(num_workers == None):
        print("Argument num_workers left blank, determining num_workers..")
        num_workers = helpers.determine_num_workers(printers=printers)

    # Carry over from MatLab
    data = data * 100

    # Cut Lambdas, divisible by 2 for MP
    # Not defining this in def: line because mutable arguments are bad style
    if(all_lambda==None): all_lambda = [20, 10, 5, 3, 2, 1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]

    # Need to fit for half frequency because we're taking half the data
    ratio = 0.5
    gamma = 1 - (1 - gamma) / ratio

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
    # partial_f = partial(convar.convar_cow, odd_traces, gamma)
    partial_f = partial(convar.convar_np, odd_traces, gamma, num_iters=convar_num_iters, early_stop_bool=early_stop_bool)
    with Pool(num_workers) as p:
        results = p.map(partial_f, all_lambda)

    end = time.time()
    if(printers): print("All Convars time:", end - start)

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

    if (printers):
        print("------------------------------------------------------")
        print("------------------------------------------------------")
        print("Min error Convar:", min_error_convar)
        print("Best Lambda:", best_lambda_convar)

    return best_lambda_convar


"""
This function chunks P to use Multiprocessing except if chunk_t_bool is True. (nochmal bisschen drüber nachdenken wie man das hier macht...) 
Argument List:
data                - input numpy ndarray TxP
    
gamma               - calcium decay rate (single neuron, based on 40Hz measurments in Gcamp6f mice)
    
best_lambda         - determined in find_best_lambda, default: 1
    
num_workers         - Number of parallel processes to be used
                    - If num_workers is left at None, helpers.determine_num_workers() will recommend a num_workers and use it. 
                    
chunk_t_bool        - Determines if T should be chunked and used in Multiprocessing
    
chunk_x_mode        - If this is set to "pct", it will chunk into a percentage of all data (or use x% of the chunks for overlap).
                    - If this is set to "flat", it will just take a flat amount of frames for chunks or overlap.
                    - Example 1:    chunk_size_mode="pct" and chunk_size=0.4, chunks will be 40%,40% and the remaining 20%
                    -               chunk_overlap_mode="pct" and chunk_overlap=0.5, take 50% of the chunks for overlap. The 20% remainder chunk will also use 50% of the 40% chunks.
                    - chunk_size shouldn't create chunks smaller than 10 frames.. 
                    - Try increasing chunk_size if you get this Exception: "ValueError: zero-size array to reduction operation minimum which has no identity"
                    
convar_mode         - "standard"|"cow"
                    - standard  = One conventional run with early Stop activated, First Difference method for output matrix init and standard LR
                    - cow       = Two runs, first adds the experimental adaptive LR, feeding the output of that into a standard convar run (hopefully instantly terminating)
                    
convar_algo         - "numpy"|"scipyBLAS"|"torchCUDA"

convar_num_iters    - number of iterations for convar to run 

"""

def deconvolve(
    data, gamma=0.97, best_lambda=1,
    num_workers=None,
    chunk_t_bool=False, chunk_size_mode="flat", chunk_size=14, chunk_overlap_mode="flat", chunk_overlap=10,
    convar_mode="standard", convar_algo="", convar_num_iters=10000
        ):
    """Docstring"""
    #Todo catch bad chunk mode settings, e.g. mode "flat" and size bigger than the input or something..

    if(num_workers == None):
        print("------------------------------------------------------")
        print("Argument num_workers left blank, determining num_workers..")
        num_workers = helpers.determine_num_workers()


    #Todo add convar_algo and convar_mode capability

    # Creating convar function with prefilled arguments (so Pool.map can be used)
    lambda_convar = partial(__convar_arg_reorganizer, gamma, best_lambda, convar_num_iters, convar_mode, convar_algo)

    if(chunk_t_bool):
        # Chunk T, MP on T

        data_list = helpers.chunk_list(data, mode=chunk_size_mode, chunk_size=chunk_size)

        with Pool(num_workers) as p:
            results = p.map(lambda_convar, data_list)
            # results = p.apply_async(__convar_spawner, range(0,10))
    else:
        # Chunk P, MP on P
        pass



def __convar_arg_reorganizer(gamma, _lambda, num_iters, convar_mode, convar_algo, data):
    """Needed for functool.partials (maybe?).. Else I'll use lambdas.."""
    return convar.convar_np(data, gamma, _lambda, num_iters=num_iters)


def __convar_spawner(x):
    """Private function that handles splitting P for Multiprocessing Pools"""
    #Todo: make pools be able to spawn in pools with global process limit
    print("Spawning..")
    # with Pool(8) as p:
        # results = p.map(__testo, range(0,, 10))