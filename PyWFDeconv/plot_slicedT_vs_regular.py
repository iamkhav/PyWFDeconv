from . import convar as convar
from . import helpers as helpers
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import h5py
import PyWFDeconv as wfd

def compare_slice_vs_regular(data):
    """
    Compare validity of data on sliced Ts vs regular.
    :return:
    """


    # Standard run
    fr, _, _ = convar.convar_np(data, 0.97, 1)

    # Chunk T this run
    chunk_size = 200
    overlap = 100
    num_loops = ceil(np.shape(data)[0] / chunk_size)

    for i in range(0, num_loops):
        if(i == 0):
            chunked_r, _, _ = convar.convar_np(data[(i * chunk_size): (i + 1) * chunk_size, :], 0.97, 1)
            continue
        temp_r, _, _ = convar.convar_np(data[(i*chunk_size) - overlap : (i+1)*chunk_size, :], 0.97, 1)
        chunked_r = np.concatenate((chunked_r, temp_r[overlap-1:]))

    print(np.shape(fr))
    print(np.shape(chunked_r))


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
    plt.plot(average_nochunk, label="No Chunk", linewidth=2, color="orange")
    plt.plot(average_chunked, label="Chunked T", linewidth=2, color="magenta")

    plt.ylabel("Mean per Frame")
    plt.xlabel("Frame")
    plt.title("Convar: Mean of Output per Frame, no Chunks vs Chunk")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

def thesis_overlap_comparison():
    h5_path = r"ExampleData/data_Jonas.hdf5"

    # Read files
    with h5py.File(h5_path, "r") as f:
        # List all groups
        print(f"Keys: {f.keys()}")

        # Get data and ROI
        df_fo = np.array(f["DF_by_FO"])
        ROI = np.array(f["ROI"])
    # Clean the data (if you suspect or know about Inf/NaN regions)
    data = helpers.CleanDFFO(df_fo, ROI=ROI)
    data = data[:2000,100:110,100]
    data = helpers.normalize_1_0(data)


    # wfd.find_best_lambda(data, gamma=0.92, early_stop_bool=True, printers=2, all_lambda=[0.00000000000001,0.0000001,0.00001,0.0001,0.001,0.01])
    # return
    thislambda = 10
    # Standard run
    # fr, _, _ = convar.convar_np(data, 0.92, thislambda)

    # Chunk T this run
    chunk_size = 400
    overlap = 100
    num_loops = ceil(np.shape(data)[0] / chunk_size)
    chunked_r = None

    # for i in range(0, num_loops):
    #     if (i == 0):
    #         chunked_r, _, _ = convar.convar_np(data[(i * chunk_size): (i + 1) * chunk_size, :], 0.92, thislambda)
    #         continue
    #     temp_r, _, _ = convar.convar_np(data[(i * chunk_size) - overlap: (i + 1) * chunk_size, :], 0.92, thislambda)
    #     chunked_r = np.concatenate((chunked_r, temp_r[overlap - 1:]))

    fr, _, _ = wfd.deconvolve(data,0.92,thislambda,num_workers=0)
    chunked_r, _, _ = wfd.deconvolve(data,0.92,thislambda,chunk_t_bool=True,chunk_size=chunk_size,chunk_overlap=overlap,num_workers=0)
    chunked_blend,_,_ = wfd.deconvolve(data, 0.92, thislambda, chunk_t_bool=True, chunk_size=chunk_size, chunk_overlap=overlap, num_workers=8,
                                      chunk_overlap_blend=30)
    chunked_steal,_,_ = wfd.deconvolve(data,0.92,thislambda,chunk_t_bool=True,chunk_size=chunk_size,chunk_overlap=overlap,num_workers=8,chunk_overlap_steal=30)

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

    average_blend = []
    for i in chunked_blend[start_Frame:]:
        average_blend.append(np.mean(i))
    average_blend = helpers.normalize_1_0(average_blend)

    average_steal = []
    for i in chunked_steal[start_Frame:]:
        average_steal.append(np.mean(i))
    average_steal = helpers.normalize_1_0(average_steal)

    # Plot showing averaged spiking rates
    plt.rcParams.update({'font.size': 13})
    plt.rcParams["figure.figsize"] = (8, 6)

    # plt.plot(average_original, label="Original", linewidth=2, alpha=0.2, color="b")
    plt.plot(average_nochunk, label="No Chunk", linewidth=2, color="orange")
    plt.plot(average_chunked, label="Chunked T", linewidth=2, color="magenta")

    plt.ylabel("Mean per Frame")
    plt.xlabel("Frame")
    plt.title("Convar: Mean of Output per Frame, no Chunks vs Chunk")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot showing difference
    diff = average_nochunk - average_chunked
    diff_blend = average_nochunk - average_blend
    diff_steal = average_nochunk - average_steal
    plt.axhline(y=0, color='m', linestyle=':')
    plt.plot(diff, label="No Blend")
    plt.plot(diff_blend, label="Blend")
    # plt.plot(diff_steal, label="Steal")
    plt.legend()
    plt.show()
    plt.close()

