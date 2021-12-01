from pathlib import Path
import scipy.io
import numpy as np

from tqdm import tqdm





### Loading Decomp

def anatomical_parcellation(DecompDataObject, dict_path=None):
    ### Loading meta data for parcellation, masks and labels for each area
    if dict_path is None: # Fallback
        dict_path = Path(__file__).parent.parent.parent/"resources"/"meta"/"anatomical.mat"
    spatials = np.asarray(scipy.io.loadmat(dict_path ,simplify_cells=True)['areaMasks'], dtype='bool')
    labels = np.asarray(scipy.io.loadmat(dict_path ,simplify_cells=True) ['areaLabels'], dtype ='str')

    # Maps and Spats have slightly different dims
    frames, _ = DecompDataObject.temporals_flat.shape
    n_svd , h, _ = DecompDataObject.spatials.shape
    n_segments , _ , w = spatials.shape

    svd_segments_bitmasks = np.broadcast_to(spatials,(n_svd,*spatials.shape)) #repeats spatials for every frame (not in memory, just simulates it by setting a stride )


    #use nanmean to use partially covered areas
    svd_segment_mean = np.zeros((n_svd,n_segments))
    svd_segment_mean = np.moveaxis([np.nanmean(DecompDataObject.spatials[:,:h,:w][svd_segments_bitmasks[:,i,:h,:w]].reshape(n_svd,-1),axis=-1) for i in range(n_segments)],-1,0)
    np.nan_to_num(svd_segment_mean,copy=False)

    new_temporals = np.tensordot(DecompDataObject.temporals_flat, svd_segment_mean, 1)
    new_spatials = spatials
    DecompDataObject.update(new_temporals,new_spatials, spatial_labels=labels)

    return DecompDataObject

