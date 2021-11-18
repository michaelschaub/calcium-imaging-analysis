from pathlib import Path
import scipy.io
import numpy as np

from tqdm import tqdm





### Loading Decomp

def anatomical_parcellation(DecompDataObject, mask_path=None):
    ### Loading MAT -> move to loading function
    if mask_path is None:
        data_path = Path(__file__).parent.parent.parent/"resources"
        mask_path = data_path/"meta"/"areaMasks.mat"
    spatials = np.asarray(scipy.io.loadmat(mask_path ,simplify_cells=True)['areaMasks'], dtype='bool')

    ### Processing (axes are switched)
    frames, _ = DecompDataObject.temporals_flat.shape
    n_svd , h, _ = DecompDataObject.spatials.shape
    spatials = np.moveaxis(spatials,-1,0)[:,:h,:] #w
    n_segments , _ , w = spatials.shape
    print(spatials.shape)
    print(DecompDataObject.temporals_flat.shape)


    #allframes_allsegments_bitmasks = np.broadcast_to(spatials,(frames,*spatials.shape)) #repeats spatials for every frame (not in memory, just simulates it by setting a stride )


    svd_segments_bitmasks = np.broadcast_to(spatials,(n_svd,*spatials.shape))


    #svd_segment_mean = np.zeros((n_svd,n_segments, h, w)) #array too large for RAM  -> iterate over segments, apply bitmap and take mean over h,w
    svd_segment_mean = np.zeros((n_svd,n_segments))

    #use nanmean to use partially covered areas
    svd_segment_mean = np.moveaxis([np.nanmean(DecompDataObject.spatials[:,:h,:w][svd_segments_bitmasks[:,i,:h,:w]].reshape(n_svd,-1),axis=-1) for i in range(n_segments)],-1,0)
    np.nan_to_num(svd_segment_mean,copy=False)
    #svd_segment_mean[svd_segment_mean == np.NAN] = 0

    new_temporals = np.tensordot(DecompDataObject.temporals_flat, svd_segment_mean, 1)
    new_spatials = spatials
    DecompDataObject.update(new_temporals,new_spatials)


    return DecompDataObject

    '''
    anatomical_segments = np.zeros((n_segments,h,w))
    anatomical_segments_mean = np.zeros((frames,n_segments))


    #anatomical_segments_mean = np.mean(np.broadcast_to(DecompDataObject.pixel[:,:,:w],(n_segments,*DecompDataObject.pixel[:,:,:w].shape))[spatials],axis=(-2,-1))

    for i in tqdm(range(frames)):
        anatomical_segments[spatials] = np.broadcast_to(DecompDataObject.pixel[i,:,:w],(n_segments,*DecompDataObject.pixel[i,:,:w].shape))[spatials]
        anatomical_segments_mean[i,:] = np.mean(anatomical_segments,axis=(1,2))
     
    
    for s in range(n_segments):
        anatomical_segments_mean[:,s] = np.mean(DecompDataObject.pixel[:,:,:][allframes_allsegments_bitmasks[:,s,:,:]],axis=(-2,-1))
        print(s)
    
    
    #anatomical_segments = [DecompDataObject.pixel[i,:,:][np.newaxis,:,:][spatials] for i in range(frames)]
    #anatomical_segments = [DecompDataObject.pixel[:,:,:][:,np.newaxis,:,:][allframes_allsegments_bitmasks]
    print(anatomical_segments_mean.shape)
    '''
