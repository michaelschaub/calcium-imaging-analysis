# -*- coding: utf-8 -*-
"""
Created on Sun May  5 22:34:05 2019

@author: Simon
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.
    
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import time
import warnings
warnings.filterwarnings("ignore")
from locanmf import LocaNMF
import os
        
def locaNMF_runWithLocalV(animal):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device='cuda'
    
    minrank = 1; maxrank = 20; # rank = how many components per brain region. Set maxrank to around 10 for regular dataset.
    rank_range = (minrank, maxrank, 1)
    min_pixels = 100 # minimum number of pixels in Allen map for it to be considered a brain region
    loc_thresh = 50 # Localization threshold, i.e. percentage of area restricted to be inside the 'Allen boundary'
    r2_thresh = 0.99 # Fraction of variance in the data to capture with LocaNMF
    trainingRange = 'allAudio'
    
    #%% load some data and allen info
    # Folder where the data is stored
    #datafolder='\\\grid-hs\\churchland_nlsas_data\\data\\BpodImager\\Animals\\mSM66\\blockData\\'
    #animal = 'mSM63'
    # basefolder = '\\\\churchlandNAS\\homes\\DOMAIN=CSHL\\smusall\\BpodImager\\'
    # basefolder ='\\\grid-hs\\churchland_nlsas_data\\data\\BpodImager\\'
    basefolder = 'X:\\RawData\\'

    # datafolder= basefolder + 'Animals\\'+ animal + '\\'
    # savefolder= basefolder + 'Animals\\'+ animal + '\\'
    datafolder= basefolder + animal + '\\'
    savefolder= basefolder + animal + '\\'
    print(animal)
    
    # Get allen mask and regions
    # regionMap = sio.loadmat(basefolder+'newRegionMap.mat')['newRegionMap']
    regionMap = sio.loadmat(basefolder+'newerRegionMap.mat')['newerRegionMap']
    a = regionMap.shape
    regionMap[:,:int(a[1]/2)] = regionMap[:,:int(a[1]/2)] * -1 #flip sign for hemisspheres
    
    # get recordings
    recs = sio.loadmat(datafolder+'SpatialDisc\\recs_'+trainingRange+'.mat')['recs']
    
    #%% run over recordings and perform locaNMF for each vc/u
    for cRec in recs:
        cRec = cRec.strip()
        print(animal)
        print(cRec)
    
        g = h5py.File(datafolder+'SpatialDisc\\'+cRec+'\\alignU.mat','r')
        U = np.array(g['U'])
        U = U.transpose((2,1,0))
        U = U[0:a[0], 0:a[1], :]
        
        # get mask by looking at NaNs in U
        valid_mask = np.isnan(np.mean(U, axis = 2)) == False
        
        # get pre-computed QR Results
        g = h5py.File(datafolder+'SpatialDisc\\'+cRec+'\\QR.mat','r')
        r = np.array(g['r'])
        q = np.array(g['q'])
        
        video_mats = (np.copy(U[valid_mask]), r)
        del U
        
        t0_global = time.time()
        t0 = time.time()
        time_ests={'data_prep':time.time() - t0}
        
        #% prepare data structure and run LocaNMF
        # region_mats[0] = [unique regions x pixels] the mask of each region
        # region_mats[1] = [unique regions x pixels] the distance penalty of each region
        # region_mats[2] = [unique regions] area code
        region_mats = LocaNMF.extract_region_metadata(valid_mask,
                                                    regionMap,
                                                    min_size=min_pixels)
        
        region_metadata = LocaNMF.RegionMetadata(region_mats[0].shape[0],
                                               region_mats[0].shape[1:],
                                               device=device)
        
        region_metadata.set(torch.from_numpy(region_mats[0].astype(np.uint8)),
                            torch.from_numpy(region_mats[1]),
                            torch.from_numpy(region_mats[2].astype(np.int64)))
        
        # Do SVD
        torch.cuda.synchronize()
        print('SVD Initialization')
        t0 = time.time()
        region_videos = LocaNMF.factor_region_videos(video_mats,
                                                   region_mats[0],
                                                   rank_range[1],
                                                   device=device)
        torch.cuda.synchronize()
        print("\'-total : %f" % (time.time() - t0))
        time_ests['svd_init'] = time.time() - t0
        
        low_rank_video = LocaNMF.LowRankVideo(
            (int(np.sum(valid_mask)),) + video_mats[1].shape, device=device
        )
        low_rank_video.set(torch.from_numpy(video_mats[0].T),
                           torch.from_numpy(video_mats[1]))
        
        
        torch.cuda.synchronize()
        print('Rank Line Search')
        t0 = time.time()
        locanmf_comps = LocaNMF.rank_linesearch(low_rank_video,
                                      region_metadata,
                                      region_videos,
                                      maxiter_rank=maxrank-minrank+1,
                                      maxiter_lambda=300,
                                      maxiter_hals=20,
                                      lambda_step=1.35,
                                      lambda_init=1e-6,
                                      loc_thresh=loc_thresh,
                                      r2_thresh=r2_thresh,
                                      rank_range=rank_range,
                                      verbose=[True, False, False],
                                      sample_prop=(1,1),
                                      device=device)
        torch.cuda.synchronize()
        print("\'-total : %f" % (time.time() - t0))
        time_ests['rank_linesearch'] = time.time() - t0
        print("Number of components : %d" % len(locanmf_comps))
        
        # Evaluate R^2
        _,r2_fit=LocaNMF.evaluate_fit_to_region(low_rank_video,
                                                   locanmf_comps,
                                                   region_metadata.support.data.sum(0),
                                                   sample_prop=(1, 1))
        print("R^2 fit on all data : %f" % r2_fit)
        
        time_ests['global_time'] = time.time()-t0_global
        
        # Assigning regions to components
        region_ranks = []; region_idx = []
        
        for rdx in torch.unique(locanmf_comps.regions.data, sorted=True):
            region_ranks.append(torch.sum(rdx == locanmf_comps.regions.data).item())
            region_idx.append(rdx.item())
        
        areas = region_metadata.labels.data[locanmf_comps.regions.data].cpu().numpy()
        
        #% Get LocaNMF spatial and temporal components
        A = locanmf_comps.spatial.data.cpu().numpy().T
        A_reshape=np.zeros((valid_mask.shape[0],valid_mask.shape[1],A.shape[1])); A_reshape.fill(np.nan)
        A_reshape[valid_mask,:]= A
        print(A_reshape.shape)
        
        q = np.array(g['q'])
        C = np.matmul(np.transpose(q),locanmf_comps.temporal.data.cpu().numpy().T).T
        print(C.shape)
        
        sio.savemat(savefolder+'SpatialDisc\\'+cRec+'\\newAC_'+str(maxrank)+'_'+str(loc_thresh)+'.mat',
                    {'C':C,
                     'A':A_reshape,
                     'lambdas':locanmf_comps.lambdas.data.cpu().numpy(),
                     'areas':areas,
                     'r2_fit':r2_fit,
                     'time_ests':time_ests,
                     'loc_thresh':loc_thresh,
                     'regionMap':regionMap,                     
                    })
        
        print("Output saved")
        # Show first component
        if __name__ != "__main__":
            plt.imshow(A_reshape[:,:,0]); plt.show()
            
        torch.cuda.empty_cache()
    
    print('Finished AC conversion for recordings in '+trainingRange+': ' + animal)
    
if __name__ == "__main__":
    locaNMF_runWithLocalV(sys.argv[1])