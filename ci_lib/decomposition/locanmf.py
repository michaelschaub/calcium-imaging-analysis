import scipy.io as sio
import numpy as np
import sys


if sys.version_info[:2] == (1,6): ##Guard locanmf
    from locanmf import LocaNMF
    import torch


import logging
LOGGER = logging.getLogger(__name__)

## [OPTIONAL] if cuda support, uncomment following lines
#import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# DEVICE='cuda'

# else, if on cpu
DEVICE='cpu'

def locaNMF(DecompDataObject, atlas_path, logger=LOGGER,
        minrank = 1, maxrank = 10,
        min_pixels = 100,
        loc_thresh = 70,
        r2_thresh = 0.99,
        nonnegative_temporal = False,
    ):
    """
    Decomposes a DecompDataObject with locaNMF with seeds based on a Brain Atlas (TODO cite)
    TODO a bit more formal
    TODO types

    :param DecompDataObject: DecompDataObject with abitrary parcellation (Usually SVD)
    :type DecompDataObject: DecompDataObject

    :param atlas_path: Path to Dict containing the Brain Atlas (TODO Specify Format)
    :type atlas_path: String or pathlib.Path or None

    :param minrank: how many components per brain region. Set maxrank to around 10 for regular dataset.
    :param maxrank: how many components per brain region. Set maxrank to around 10 for regular dataset.
    :param min_pixels: minimum number of pixels in Allen map for it to be considered a brain region
    :param loc_thresh: Localization threshold, i.e. percentage of area restricted to be inside the 'Allen boundary
    :param r2_thresh: Fraction of variance in the data to capture with LocaNMF
    :param nonnegative_temporal: Do you want nonnegative temporal components? The data itself should also be nonnegative in this case.

    :param logger: The LOGGER object that all console outputs are piped into
    :type logger: LOGGER

    :return: DecompDataObject with Data-driven Spatials constrained to the Brain Regions in the Atlas.
    :rtype: DecompDataObject
    """
    rank_range = (minrank, maxrank, 1)

    atlas_file = sio.loadmat(atlas_path,simplify_cells=True)
    brainmask = np.asarray(atlas_file['cortexMask'], dtype='bool')
    atlas_msk = np.asarray(atlas_file['areaMasks'], dtype='bool')
    labels = np.asarray(atlas_file['areaLabels_wSide'],dtype=str)

    temporals = DecompDataObject.temporals_flat
    spatials = np.moveaxis(DecompDataObject.spatials, 0, -1 )

    width = min(DecompDataObject.n_xaxis, brainmask.shape[0])
    height = min(DecompDataObject.n_yaxis, brainmask.shape[1])
    brainmask = brainmask[:width,:height]
    atlas_msk = atlas_msk[:,:width,:height]
    spatials = spatials[:width,:height,:]
    nan_mask = np.logical_not(np.isnan(spatials).any(axis=-1))
    atlas_msk &= nan_mask[None,:,:]
    brainmask &= nan_mask

    atlas = np.zeros((width,height), dtype=float)
    for i,a in enumerate(atlas_msk):
        atlas[a] = i

    # region_mats[0] = [unique regions x pixels] the mask of each region
    # region_mats[1] = [unique regions x pixels] the distance penalty of each region
    # region_mats[2] = [unique regions] area code
    region_mats = LocaNMF.extract_region_metadata(brainmask, atlas, min_size=min_pixels)

    region_metadata = LocaNMF.RegionMetadata(region_mats[0].shape[0],
                                            region_mats[0].shape[1:],
                                            device=DEVICE)

    region_metadata.set(torch.from_numpy(region_mats[0].astype(np.uint8)),
                        torch.from_numpy(region_mats[1]),
                        torch.from_numpy(region_mats[2].astype(np.int64)))

    # Perform the LQ decomposition. Time everything.
    if nonnegative_temporal:
        r = temporals
    else:
        q, r = np.linalg.qr(temporals)

    # Put in data structure for LocaNMF
    video_mats = (np.copy(spatials[brainmask]), r.T)
    # Do SVD
    region_videos = LocaNMF.factor_region_videos(video_mats,
                                                region_mats[0],
                                                rank_range[1],
                                                device=DEVICE)

    low_rank_video = LocaNMF.LowRankVideo( (int(np.sum(brainmask)),) + video_mats[1].shape,
                                            device=DEVICE )

    low_rank_video.set(torch.from_numpy(video_mats[0].T),
                        torch.from_numpy(video_mats[1]))

    locanmf_comps = LocaNMF.rank_linesearch(low_rank_video,
                                            region_metadata,
                                            region_videos,
                                            maxiter_rank=maxrank-minrank+1,
                                            maxiter_lambda=20,
                                            maxiter_hals=20,
                                            lambda_step=1.35,
                                            lambda_init=1e-6,
                                            loc_thresh=loc_thresh,
                                            r2_thresh=r2_thresh,
                                            rank_range=rank_range,
                                            nnt=nonnegative_temporal,
                                            verbose=[True, False, False],
                                            sample_prop=(1,1),
                                            device=DEVICE
                                            )
    # Evaluate R^2
    #_, r2_fit = LocaNMF.evaluate_fit_to_region(low_rank_video,
                                               #locanmf_comps,
                                               #region_metadata.support.data.sum(0),
                                               #sample_prop=(1, 1))

    # Get LocaNMF spatial and temporal components
    A=locanmf_comps.spatial.data.cpu().numpy().T
    A_pixel=np.zeros((width,height,A.shape[1]))
    A_pixel.fill(np.nan)
    A_pixel[brainmask,:] = A

    if nonnegative_temporal:
        C = locanmf_comps.temporal.data.cpu().numpy().T
    else:
        C = np.matmul(q,locanmf_comps.temporal.data.cpu().numpy().T)

    # (X,Y,K) -> (K, X, Y)
    new_spatials = np.moveaxis(A_pixel, -1, 0)
    # (T,K)
    new_temporals = C

    regions = locanmf_comps.regions.data.cpu().numpy()
    max_len = max([len(l) for l in labels])
    n_region = [ (regions == i).sum() for i in range(len(labels))]
    max_len += 1 + len(str(max(n_region)))
    new_labels = np.empty_like(regions, dtype="<U{}".format(max_len))
    logger.debug("1_region {}".format((regions == 1)))
    logger.debug("n_region {}".format(n_region))
    i_region = np.zeros_like(labels, dtype=int)
    for i,r in enumerate(regions):
        if n_region[r] == 1:
            new_labels[i] = labels[r]
        else:
            new_labels[i] = "{}#{}".format(labels[r],i_region[r])
            logger.debug(f"{labels[r]}#{i_region[r]}")
            i_region[r] += 1
    logger.debug("dtype new_labels {}".format(new_labels.dtype))
    logger.debug("shape new_labels {}".format(new_labels.shape))
    logger.debug("new_labels {}".format(new_labels))


    return DecompDataObject.update(new_temporals, new_spatials, spatial_labels=new_labels)
