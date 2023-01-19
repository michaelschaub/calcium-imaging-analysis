import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.io
from pathlib import Path

import logging
LOGGER = logging.getLogger(__name__)

from .utils import polygons_from_mask,plt_polygons

##Assumes that spatial is identical for all given temps

#def draw_neural_activity(temps,spatials,plt_title,subfig_titles):
#    pass

def plot_spatial_activity(activity,decomp_object,overlay=False):
    pass

def draw_neural_activity(frames,path=None,plt_title="",subfig_titles=None,overlay=True,outlined=True,masked=True,logger=LOGGER,vmin=None,vmax=None,font_scale=1):
    """ Draws multiple frames of neural activity in the spatial context of the brain with optional Atlas-Overlay and Cutout.

    Args:
        frames (_type_): _description_
        path (_type_, optional): _description_. Defaults to None.
        plt_title (str, optional): _description_. Defaults to "".
        subfig_titles (_type_, optional): _description_. Defaults to None.
        overlay (bool, optional): _description_. Defaults to False.
        cortex_map (bool, optional): _description_. Defaults to False.
        logger (_type_, optional): _description_. Defaults to LOGGER.
        vmin (_type_, optional): _description_. Defaults to None.
        vmax (_type_, optional): _description_. Defaults to None.
    """    

    #Single Frame is wrapped
    frames=np.asarray(frames,dtype=float)
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]
        subfig_titles = [""]
    
    if subfig_titles is None:
        n_digits = math.floor(math.log(len(frames), 10))
        subfig_titles = [str(i).zfill(n_digits ) for i in range(len(frames))]

    if overlay:
        #Hardcoded for now
        atlas_path = Path(__file__).parent.parent.parent/"resources"/"meta"/"anatomical.mat"
        #edge_map = scipy.io.loadmat(atlas_path ,simplify_cells=True)['edgeMap'] #TODO polygons instead of edge map
        #edge_map_masked =np.ma.masked_where(edge_map < 1, edge_map)


        area_masks = scipy.io.loadmat(atlas_path ,simplify_cells=True)['areaMasks']
        region_outlines = polygons_from_mask(area_masks,type="polygon")

    if outlined or masked:
        atlas_path = Path(__file__).parent.parent.parent/"resources"/"meta"/"anatomical.mat"
        cortex_mask = scipy.io.loadmat(atlas_path ,simplify_cells=True)['cortexMask']
        #cortex_mask = np.ma.masked_where(cortex_mask < 1, cortex_mask)
        mask_h,mask_w = cortex_mask.shape

        if outlined:
            outline = polygons_from_mask(cortex_mask,type="polygon")

    _ , h, w = frames.shape
    

    #Indices of subplots
    y_dims = int(np.ceil(np.sqrt(len(frames))))
    x_dims = int(np.ceil(len(frames) / y_dims))
    logger.info(f"x_dim {x_dims} y_dim {y_dims}")

    #
    mpl.rcParams.update({'font.size': 10*font_scale})
    fig, ax = plt.subplots(x_dims , y_dims, squeeze=False, constrained_layout=True)
    fig.suptitle(plt_title,y=1.02)
    plt.subplots_adjust(wspace=0)

    #Uniform vmax,vmin over all subfigures, diverging color map centered at 0, scales ind. to both sides, if one sign is larger by magnitude 2, cut off that sign
    vmin, vmax = (np.nanmin(frames) if vmin is None else vmin,np.nanmax(frames) if vmax is None else vmax)
    logger.info(f"vmin {vmin},vmax {vmax}")
    #if vmax>100*-vmin:
    #    vmin = -vmax
    #if -vmin>100*vmax:
    #    vmax = vmin
    #frames[frames<vmin]=0
    #frames[frames>vmax]=0
    #vmin = -1
    #vmax = 1

    vmin,vmax = (np.amin([vmin,-vmax]),np.amax([vmax,-vmin])) #vmin is too close to 0, 0 values will be plotted with a value != 0 due to floating point precision
    #logger.info(f"vmin,vmax {(np.nanmin(frames) if vmin is None else vmin,np.nanmax(frames) if vmax is None else vmax)}")

    #cmap = shiftedColorMap(mpl.cm.get_cmap('seismic'),vcenter=(vmin+vmax)/2)
    #cmap=mpl.cm.get_cmap('seismic')

    for j in range(y_dims):
        for i in range(x_dims):
            if j*x_dims + i < len(frames):
                #frame =  np.tensordot(temps[], spatial, 1) #np.einsum( "n,nij->ij", temps[h*width + w], spatial) #np.tensordot(temps[w + h], spatial, (-1, 0)) #np.dot(spatial,temps[w*height + h]) #
                frame = np.asarray(frames[j*x_dims + i],dtype=float)
                if masked:                   
                    frame[:mask_h,:mask_w][cortex_mask[:h,:w]==0] =  np.nan   # = np.ma.masked_where(cortex_mask == 0,frame)
                    frame = frame[:mask_h,:mask_w]

                logger.info(f"vmin {vmin},vmax {vmax}")
                im = ax[i, j].imshow(frame,cmap="seismic",norm=mpl.colors.TwoSlopeNorm(vcenter=0,vmin=vmin if vmin<0 else None,vmax=vmax if vmax>0 else None))


                if overlay:
                    plt_polygons(ax[i, j],region_outlines ,edgecolor=(1,1,1,0.8),fill=False,linewidth=0.5)

                if outlined:
                    plt_polygons(ax[i, j],outline,edgecolor="black",fill=False,linewidth=2) #facecolor=None,

                ax[i, j].set_title(subfig_titles[j*x_dims + i])
                ax[i, j].axis('off')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
    
    #fig.subplots_adjust(right=0.85)
    #cbar_ax = fig.add_axes([0.85, 0.25, 0.03, 0.5])
    fig.colorbar(ax[0,0].get_images()[0], ax=ax, shrink=0.6)
    #fig.colorbar(ax[0,0].get_images()[0], cax=cbar_ax)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()
