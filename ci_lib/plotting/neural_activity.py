import matplotlib.pyplot as plt
import numpy as np
import math

#For Brain Atlas
import scipy.io
from pathlib import Path

import logging
LOGGER = logging.getLogger(__name__)


##Assumes that spatial is identical for all given temps

#def draw_neural_activity(temps,spatials,plt_title,subfig_titles):
#    pass

def draw_neural_activity(frames,path,plt_title,subfig_titles=None,overlay=False,cortex_map=False,logger=LOGGER,vmin=None,vmax=None):
    #Single Frame is wrapped
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]
        subfig_titles = [subfig_titles]

    if subfig_titles is None:
        n_digits = math.floor(math.log(len(frames), 10))
        subfig_titles = [str(i).zfill(n_digits ) for i in range(len(frames))]

    if overlay:
        #Hardcoded for now
        atlas_path = Path(__file__).parent.parent.parent/"resources"/"meta"/"anatomical.mat"
        edge_map = scipy.io.loadmat(atlas_path ,simplify_cells=True)['edgeMap']
        edge_map_masked =np.ma.masked_where(edge_map < 1, edge_map)

    if cortex_map:
        atlas_path = Path(__file__).parent.parent.parent/"resources"/"meta"/"anatomical.mat"
        cortex_map = scipy.io.loadmat(atlas_path ,simplify_cells=True)['cortexMask']



    _ , h, w = frames.shape

    #Indices of subplots
    y_dims = int(np.ceil(np.sqrt(len(frames))))
    x_dims = int(np.ceil(len(frames) / y_dims))

    logger.info(f"x_dim {x_dims} y_dim {y_dims}")

    fig, ax = plt.subplots(x_dims , y_dims, constrained_layout=True, squeeze=False)
    fig.suptitle(plt_title)
    for j in range(y_dims):
        for i in range(x_dims):
            if j*x_dims + i < len(frames):
                #frame =  np.tensordot(temps[], spatial, 1) #np.einsum( "n,nij->ij", temps[h*width + w], spatial) #np.tensordot(temps[w + h], spatial, (-1, 0)) #np.dot(spatial,temps[w*height + h]) #
                frame = frames[j*x_dims + i]
                if(cortex_map):
                    frame[cortex_map[:h,:w]==0] = np.nan
                im = ax[i, j].imshow(frame,vmin=vmin, vmax=vmax)
                if overlay:
                    ax[i, j].imshow(edge_map_masked[:h,:w], interpolation='none')

                fig.colorbar(im, ax=ax[i, j])
                ax[i, j].set_title(subfig_titles[j*x_dims + i])
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                #plt.draw()
                #plt.pause(0.1)
    #plt.show()
    plt.savefig(path, format='png')
    plt.close()
