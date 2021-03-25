from torch.multiprocessing import Pool
from . import (
    helpers
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
data            - input numpy ndarray TxP

gamma           - has to have the ratio calculation if even/odd mode exists
                - calcium decay rate (single neuron, based on 40Hz measurments in Gcamp6f mice)
                
range_pct       - default: 1.0

num_workers     - If num_workers is left at None, helpers.determine_num_workers() will recommend a num_workers and use it.

all_lambda      - Lambdas to be tested
"""


"""
Deprecated features:
mode            - "evenodd" for even/odd cross validation method
                - "range" for taking only first range_pct of data           ~~~Jonas meint eher unsinnvoll, Todo testen
                - "evenoddrange" for even/odd cross validation method while taking only first range_pct of data
                - default: "evenodd"
                
range_pct       - default: 0.1
"""


def find_best_lambda(data, gamma=0.97,  range_pct=1.0, num_workers=None, all_lambda=None):
    # Workers Init
    if(num_workers == None):
        print("Argument num_workers left blank, determining num_workers..")
        num_workers = helpers.determine_num_workers()



    # Original Lambdas
    # if(all_lambda==None): all_lambda = [80, 40, 20, 10, 7, 5, 3, 2, 1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]

    # Carry over from MatLab
    data = data * 100

    # Cut Lambdas, divisible by 2 for MP
    if(all_lambda==None): all_lambda = [20, 10, 5, 3, 2, 1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]

    # Note: If needed I could program methods for
    # Need to fit for 20hz because we're taking half the data
    ratio = 0.5
    gamma = 1 - (1 - gamma) / ratio

    odd_traces = data[0::2]
    even_traces = data[1::2]





def deconvolve():
    pass



"""
check t slicing:

slice size
slice overlap
 

"""