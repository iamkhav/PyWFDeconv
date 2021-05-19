from . import convar as convar
from . import helpers as helpers
import numpy as np
import matplotlib.pyplot as plt
from math import ceil


def plot_r_finals(r_final_list):
    start_Frame = 10

    # Plot 1
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["figure.figsize"] = (8, 6)

    c = 1
    for r in r_final_list:
        mean_per_frame = []
        for i in r[start_Frame:]:
            mean_per_frame.append(np.mean(i))
        mean_per_frame = helpers.normalize_1_0(mean_per_frame)
        plt.plot(mean_per_frame, label=c, linewidth=2)
        c += 1

    plt.ylabel("Mean per Frame")
    plt.xlabel("Frame")
    plt.title("Convar: Mean of Output per Frame")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()