from pathlib import Path
import scipy.io
import numpy as np
from sklearn.decomposition import FastICA

import logging
LOGGER = logging.getLogger(__name__)

def anatomical_parcellation(DecompDataObject, atlas_path=None, ROI=[], logger=LOGGER, ):
    '''
    Decomposes a DecompDataObject into a anatomical parcellation based on a Brain Atlas

    :param DecompDataObject: DecompDataObject with abitrary parcellation (Usually SVD)
    :type DecompDataObject: DecompDataObject

    :param atlas_path: Path to Dict containing the Brain Atlas (TODO Specify Format)
    :type atlas_path: String or pathlib.Path or None

    :param ROI: Regions of interest to use
    :type ROI: [Strings] or None

    :param logger: The LOGGER object that all console outputs are piped into
    :type logger: LOGGER

    :return: Anatomically parcellated DecompDataObject with Spatials corresponding to the given Atlas.
    :rtype: DecompDataObject
    '''

    ### Loading meta data for parcellation, masks and labels for each area
    if atlas_path is None: # Fallback
        atlas_path = Path(__file__).parent.parent.parent/"resources"/"meta"/"anatomical.mat"
    spatials = np.asarray(scipy.io.loadmat(atlas_path ,simplify_cells=True)['areaMasks'], dtype='bool')
    labels = np.asarray(scipy.io.loadmat(atlas_path ,simplify_cells=True) ['areaLabels_wSide'],dtype=str)


    #To load only ROI from of anatomical atlas
    if ROI != [] and ROI !='':
        if isinstance(ROI,str):
            ROI = ROI.split(',')
        else:
            ROI = ROI[0].split(',')

        for i,region in enumerate(ROI):
            if "-R" in ROI[i] or "-L" in ROI[i]:
                ROI[i]= ROI[i].replace("-R","ᴿ")
                ROI[i]= ROI[i].replace("-L","ᴸ")
            #else:
            #    ROI[i]= ROI[i]+"ᴿ"
            #    np.insert(ROI,i,ROI[i]+"ᴸ")

        #print([region for region in ROI if region in labels])
        ind = [i for region in ROI for i, label in enumerate(labels) if region in label]

        #iter=[(spatials[i],region) for i,region in enumerate(labels) if region in ROI]
        if ind!=[]:
            spatials, labels = spatials[ind,:,:],labels[ind]
        else:
            logger.warn("No ROI matched Atlas from "+",".join[ROI])


    #Filter according to labels
    #if filter_labels is not None:
    #    pass

    # Maps and Spats have slightly different dims
    frames, _ = DecompDataObject.temporals_flat.shape
    n_svd , h, _ = DecompDataObject.spatials.shape
    n_segments , _ , w = spatials.shape

    svd_segments_bitmasks = np.broadcast_to(spatials,(n_svd,*spatials.shape)) #repeats spatials for every frame (not in memory, just simulates it by setting a stride )

    svd_segment_mean = np.zeros((n_svd,n_segments))
    svd_segment_mean = np.moveaxis([np.nanmean(DecompDataObject.spatials[:,:h,:w][svd_segments_bitmasks[:,i,:h,:w]].reshape(n_svd,-1),axis=-1) for i in range(n_segments)],-1,0)
    np.nan_to_num(svd_segment_mean,copy=False)

    new_temporals = np.tensordot(DecompDataObject.temporals_flat, svd_segment_mean, 1)
    new_spatials = spatials

    return DecompDataObject.update(new_temporals,new_spatials, spatial_labels=labels)

def fastICA(DecompDataObject, n_comps):
    """
    Decomposes an DecompDataObject with Independet Component Analysis

    :param DecompDataObject: DecompDataObject with abitrary parcellation (Usually SVD)
    :type DecompDataObject: DecompDataObject

    :param n_comps: Number of independent components (w.r.t. time).
    :type n_comps: int

    :return: DecompDataObject with Spatials corresponding to the independent components (w.r.t. time) obtianed by ICA.
    :rtype: DecompDataObject
    """
    #Eventually add mask?

    ica = FastICA(n_components=n_comps,
                  random_state=0)

    new_temporals = ica.fit_transform(DecompDataObject.temporals_flat)

    inverse = ica.mixing_.T #    inverse = ica.mixing_
    new_spatials = np.tensordot(inverse, DecompDataObject.spatials, axes=1)

    return DecompDataObject.update(new_temporals, new_spatials)
