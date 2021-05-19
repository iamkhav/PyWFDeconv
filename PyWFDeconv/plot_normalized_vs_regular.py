from . import convar as convar
from . import helpers as helpers
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def compare_normalized_vs_regular(data):
    """
    Compare validity of data on sliced Ts vs regular.
    :return:
    """


    # Standard run
    fr, _, _ = convar.convar_np(data, 0.97, 20, early_stop_bool=False, num_iters=500)

    chunked_r, _, _ = convar.convar_np(helpers.normalize_1_0(data), 0.97, 20, early_stop_bool=False, num_iters=500)


    start_Frame = 10


    average_original = []
    for i in data[start_Frame:]:
        # average_original.append(np.mean(i) / np.std(y))
        average_original.append(np.mean(i))
    average_original = helpers.normalize_1_0(average_original)

    average_nochunk = []
    for i in fr[start_Frame:]:
        average_nochunk.append(np.mean(i))
    average_nochunk = helpers.normalize_1_0(average_nochunk)


    average_chunked = []
    for i in chunked_r[start_Frame:]:
        average_chunked.append(np.mean(i))
    average_chunked = helpers.normalize_1_0(average_chunked)


    # Plot 1
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["figure.figsize"] = (8, 6)

    plt.plot(average_original, label="Original", linewidth=2, alpha=0.2, color="b")
    plt.plot(average_nochunk, label="Regular Input Data", linewidth=2, color="orange")
    plt.plot(average_chunked, label="Normalized Input Data", linewidth=2, color="magenta")

    plt.ylabel("Mean per Frame")
    plt.xlabel("Frame")
    plt.title("Convar: Mean of Output per Frame, Normalized[0,1] vs Regular")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()