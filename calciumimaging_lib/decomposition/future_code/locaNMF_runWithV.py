# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:15:03 2019

@author: Simon
"""

import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import torch
import LsNMF
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Folder where the data is stored
datafolder='C:\\Demo\\'


# File 'Vc_Uc.mat' in the above folder contains Uc, Vc, brainmask_aligned
arrays={}
g = h5py.File(datafolder+'nVc_Uc.mat','r')
for k, v in g.items():
    print(k)
    arrays[k] = np.array(v)
    
# Get data in the correct format
new_x, new_y = 580, 540 # New x and y
V=arrays['Vc'].T
U=arrays['Uc'].transpose((2,1,0))[:new_y, :new_x, :]
brainmask=~np.isnan(arrays['brainmask_aligned'].T[:new_y, :new_x])
brainmask_full=~np.isnan(arrays['brainmask_aligned'].T)
del arrays

# Check that data has the correct shapes. V [K_d x T], U [X x Y x K_d], brainmask [X x Y]
print(V.shape); print(U.shape); print(brainmask.shape)

# Get region based Allen map
dorsalMapScaled = sio.loadmat('C:\\Users\\smusall\\Documents\\repoland\\playgrounds\\Simon\\dorsalMapScaled.mat')['dorsalMapScaled']
dorsalMapScaled[:,:int(dorsalMapScaled.shape[1]/2)] = dorsalMapScaled[:,:int(dorsalMapScaled.shape[1]/2)] * -1
dorsalMapScaled = dorsalMapScaled[:new_y, :new_x]

# Perform the LQ decomposition. Time everything.
t0_global = time.time()
t0 = time.time()
q, r = np.linalg.qr(V.T)
time_ests={'qr_decomp':time.time() - t0}

# Put in data structure for LocaNMF
video_mats = (np.copy(U[brainmask]), r.T)
del U

valid_mask = brainmask
region_map = dorsalMapScaled
rank_range = (1, 14, 1) # minimum rank 1, maximum rank 14 for every region
device='cuda'

# region_mats[0] = [unique regions x pixels] the mask of each region
# region_mats[1] = [unique regions x pixels] the distance penalty of each region
# region_mats[2] = [unique regions] area code
region_mats = LsNMF.extract_region_metadata(valid_mask,
                                            region_map,
                                            min_size=rank_range[1])

region_metadata = LsNMF.RegionMetadata(region_mats[0].shape[0],
                                       region_mats[0].shape[1:],
                                       device=device)

region_metadata.set(torch.from_numpy(region_mats[0].astype(np.uint8)),
                    torch.from_numpy(region_mats[1]),
                    torch.from_numpy(region_mats[2].astype(np.int64)))


# Do SVD
torch.cuda.synchronize()
print('SVD Initialization')
t0 = time.time()
region_videos = LsNMF.factor_region_videos(video_mats,
                                           region_mats[0],
                                           rank_range[1])
torch.cuda.synchronize()
print("\'-total : %f" % (time.time() - t0))
time_ests['svd_init'] = time.time() - t0


low_rank_video = LsNMF.LowRankVideo(
    (int(np.sum(valid_mask)),) + video_mats[1].shape, device=device
)
low_rank_video.set(torch.from_numpy(video_mats[0].T),
                   torch.from_numpy(video_mats[1]))

loc_thresh=80
r2_thresh=0.99
torch.cuda.synchronize()
print('Rank Line Search')
t0 = time.time()
debug = LsNMF.rank_linesearch(low_rank_video,
                              region_metadata,
                              region_videos,
                              maxiter_rank=50,
                              maxiter_lambda=300,
                              maxiter_hals=20,
                              lambda_step=1.35,
                              lambda_init=1e-6,
                              loc_thresh=loc_thresh,
                              r2_thresh=r2_thresh,
                              rank_range=rank_range,
                              verbose=[True, False, False],
                              sample_prop=(1,1))
torch.cuda.synchronize()
print("\'-total : %f" % (time.time() - t0))
time_ests['rank_linesearch'] = time.time() - t0
print("Number of components : %f" % len(debug))

# Evaluate R^2
_,r2_fit=LsNMF.evaluate_fit_to_region(low_rank_video,debug,region_metadata.support.data.sum(0),sample_prop=(1, 1))
print("R^2 fit on all data : %f" % r2_fit)
time_ests['global_time'] = time.time()-t0_global

# Assigning regions to components
region_ranks = []
for rdx in torch.unique(debug.regions.data):
    region_ranks.append(torch.sum(rdx == debug.regions.data).item())
areas=np.repeat(region_mats[2],region_ranks,axis=0)

# Plotting all the regions' spatial map
region_ranks = [0]
region_idx = []
for rdx in torch.unique(debug.regions.data):
    region_ranks.append(torch.sum(rdx == debug.regions.data).item())
    region_idx.append(rdx.item())
    
A = np.zeros(valid_mask.shape, dtype=np.float32)
for rdx, i in zip(region_idx, np.cumsum(region_ranks[:-1])):
    fig, axs = plt.subplots(1 + int(region_ranks[1+rdx] / 4), 4,
                            figsize=(16,(1 + int(region_ranks[1+rdx] / 4)) * 4))
    axs = axs.reshape((int(np.prod(axs.shape)),))
    A[valid_mask] = debug.distance.data[i].cpu()==0
    axs[0].imshow(A)
    axs[0].set_title("Region: {}".format(rdx+1))
    for j, ax in enumerate(axs[1:]):
        if i + j < len(debug) and debug.regions.data[i+j].item() == rdx:
            A[valid_mask] = debug.spatial.data[i+j].cpu()
            ax.set_title("Component {}".format(i+j))
        else:
            A[valid_mask] = 0
            ax.imshow(A)
        
#    plt.show()
    
# Plot the distribution of lambdas. 
# If lots of values close to the minimum, decrease lambda_init.
# If lots of values close to the maximum, increase maxiter_lambda or lambda_step.
plt.hist(np.log(debug.lambdas.data.cpu()), bins=torch.unique(debug.lambdas.data).shape[0])
#plt.show()            

C=np.matmul(q,debug.temporal.data.cpu().numpy().T).T
A=debug.spatial.data.cpu().numpy().T
A_reshape=np.zeros((brainmask_full.shape[0],brainmask_full.shape[1],A.shape[1])); A_reshape.fill(np.nan)
A_reshape[brainmask_full,:]=A

# Plot first component
plt.imshow(A_reshape[:,:,0]); 
#plt.show()
sio.savemat(datafolder+'locanmf_decomp_loc'+str(loc_thresh)+'.mat',
            {'C':C,
             'A':A_reshape,
             'lambdas':debug.lambdas.data.cpu().numpy(),
             'areas':areas,
             'r2_fit':r2_fit,
             'time_ests':time_ests
            })
    
torch.cuda.empty_cache()