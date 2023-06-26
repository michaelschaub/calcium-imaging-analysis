'''
This module contains functions for calculating variously decomposed DecompData
from SVD decomposed DecompData objects
'''

# pylint: disable=invalid-name

import logging
from pathlib import Path
import scipy.io
import numpy as np
from sklearn.decomposition import FastICA

LOGGER = logging.getLogger(__name__)

def anatomical_parcellation(data, atlas_path=None, ROI=None, logger=LOGGER, ):
    '''
    Decomposes a DecompData object into a anatomical parcellation based on a Brain Atlas

    :param data: DecompData object with abitrary parcellation (Usually SVD)
    :type data: DecompData

    :param atlas_path: Path to Dict containing the Brain Atlas (TODO Specify Format)
    :type atlas_path: String or pathlib.Path or None

    :param ROI: Regions of interest to use
    :type ROI: [Strings] or None

    :param logger: The LOGGER object that all console outputs are piped into
    :type logger: LOGGER

    :return: Anatomically parcellated DecompData object with Spatials corresponding
             to the given Atlas.
    :rtype: DecompData
    '''

    ### Loading meta data for parcellation, masks and labels for each area
    if atlas_path is None: # Fallback
        atlas_path = Path(__file__).parent.parent.parent/"resources"/"meta"/"anatomical.mat"
    spatials = np.asarray(
            scipy.io.loadmat(atlas_path ,simplify_cells=True)['areaMasks'], dtype='bool')
    labels = np.asarray(
            scipy.io.loadmat(atlas_path ,simplify_cells=True) ['areaLabels_wSide'], dtype=str)


    #To load only ROI from of anatomical atlas
    if ROI is not None and ROI != [] and ROI !='':
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

        ind = [i for region in ROI for i, label in enumerate(labels) if region in label]

        if ind!=[]:
            spatials, labels = spatials[ind,:,:],labels[ind]
        else:
            logger.warning("No ROI matched Atlas from %s", ",".join(ROI))


    #Filter according to labels
    #if filter_labels is not None:
    #    pass

    # Maps and Spats have slightly different dims
    n_svd , height, _ = data.spatials.shape
    n_segments , _ , width = spatials.shape

    #repeats spatials for every frame (not in memory, just simulates it by setting a stride )
    svd_segments_bitmasks = np.broadcast_to(spatials,(n_svd,*spatials.shape))

    svd_segment_mean = np.zeros((n_svd,n_segments))

    svd_spatials = data.spatials

    #Add random noise to avoid Var=0 for empty areas
    svd_spatials = ( np.nan_to_num(svd_spatials)
                    + np.random.rand(*svd_spatials.shape) * 16 * np.finfo(np.float32).eps)

    svd_segment_mean = np.moveaxis(
            [ np.nanmean(
                (
                    svd_spatials[:,:height,:width][svd_segments_bitmasks[:,i,:height,:width]]
                ).reshape(n_svd,-1),
                axis=-1)
             for i in range(n_segments)], -1, 0)
    np.nan_to_num(svd_segment_mean,copy=False)

    new_temporals = np.tensordot(data.temporals_flat, svd_segment_mean, 1)
    new_spatials = spatials

    logger.info("NaNs in spatials: %s, NaNs in temporals: %s",
                np.isnan(new_spatials).any(),
                np.isnan(new_temporals).any())
    logger.info("0 Vars in spatials: %s, 0 Vars in temporals: %s",
                np.count_nonzero(np.var(new_spatials,axis=0)),
                np.count_nonzero(np.var(new_temporals,axis=0)==0))

    return data.recreate(new_temporals,new_spatials, spatial_labels=labels)

def fastICA(data, n_components):
    """
    Decomposes an DecompData object with Independet Component Analysis

    :param data: DecompData object with abitrary parcellation (Usually SVD)
    :type data: DecompData

    :param n_components: Number of independent components (w.r.t. time).
    :type n_components: int

    :return: DecompData object with Spatials corresponding to the independent
             components (w.r.t. time) obtained by ICA.
    :rtype: DecompData
    """
    #Eventually add mask?

    ica = FastICA(n_components=n_components,
                  random_state=0)

    new_temporals = ica.fit_transform(data.temporals_flat)

    inverse = ica.mixing_.T #    inverse = ica.mixing_
    new_spatials = np.tensordot(inverse, data.spatials, axes=1)

    return data.recreate(new_temporals, new_spatials)

def postprocess_SVD(data, n_components):
    """
    Modifies an SVD DecompData object, for example to crop to the number of components

    :param data: DecompData object with SVD parcellation
    :type data: DecompData

    :param n_components: Number of components (w.r.t. time).
    :type n_components: int

    :return: DecompData object containing the processes SVD
    :rtype: DecompData
    """

    return data.recreate(data.temporals_flat[:,:n_components], data.spatials[:n_components,:,:])
