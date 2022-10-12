import matplotlib.pyplot as plt
import numpy as np

def plot_connectivity_matrix(matrices,title,path=None):
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subplots(1, len(matrices), sharey=True)

    #fig, ax = plt.subplots(, )
    fig.suptitle(title)

    for i,matrix in enumerate(matrices):
        sub = subfigs[i].imshow(matrix)
        plt.colorbar(sub)


    fig.savefig(path, format='png')
    plt.close()