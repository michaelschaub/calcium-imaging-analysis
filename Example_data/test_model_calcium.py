#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

# libraries

import numpy as np
import h5py
import matplotlib.pyplot as plt

# import scipy.linalg as spl
# import scipy.stats as stt



#%% load signals

file_path = r'GN06/2021-01-20_10-15-16/SVD_data/Vc.mat'
arrays = {}
f = h5py.File(file_path, 'r')

#frameCnt = np.array(f['frameCnt'])

# currently there is a 2sec (@15Hz -> 30frames) baseline before the stimulus. This information is included in the task_data
select_frames = range(3000,6000) # 97450 in total

Vc = np.array(f['Vc'][select_frames, :]) # transformed signal (time, component)
U = np.array(f['U']) # mixing matrix (component x voxel x voxel)

n_T = Vc.shape[0]
C = U.shape[0] # number of components
M = U.shape[1] # vertical voxels
N = U.shape[2] # horizontal voxels


#%% calculate autocovariance

n_tau = 10 # number of lags

# Lag (time-shifted) FC matrices
cov_Vc = np.zeros([n_tau, C, C], dtype=np.float)
# Remove mean in the time series
dm_Vc = Vc - Vc.mean(axis=0)
# Calculate the lagged FC matrices
n_T_span = n_T - n_tau + 1
for i_tau in range(n_tau):
    cov_Vc[i_tau,:,:] = np.tensordot(dm_Vc[0:n_T_span,:], \
                                     dm_Vc[i_tau:n_T_span+i_tau,:], \
                                     axes=(0,0)) 
cov_Vc /= float(n_T_span - 1)
ac = cov_Vc.diagonal(axis1=1, axis2=2)


# plot autocovariance
plt.figure()
plt.plot(range(n_tau), ac)
plt.title('autocov')


# plot log autocovariance
plt.figure()
plt.plot(range(n_tau), np.log(ac))
plt.title('log autocov')


# calculate and plot time constants of exponential decay
tc = np.zeros([C])
n_fit = 4 # number of time lags in the 
for c in range(C):
    # rectify autocovariance
    ac_tmp = np.maximum(ac[:n_fit,c], 1e-10)
    # fit to estimate exponential decay
    tc[c] = -1.0 / np.polyfit(range(n_fit), np.log(ac_tmp), 1)[0]

plt.figure()
plt.hist(tc, bins=50)
plt.title('time constants')


# plot variance of components
plt.figure()
plt.plot(range(C), ac[0,:])
plt.title('variance of components')


#%% 

# project in original space
def bproj(x):
    return np.tensordot(x, U, axes=(-1,0))

# mean over time
m_Vc = Vc.mean(axis=0)
# variance over time
var_Vc = Vc.var(axis=0)

# plot mean
plt.figure()
plt.imshow(bproj(m_Vc), vmin=-0.0004, vmax=0.0004)
plt.colorbar()
plt.title('mean profile')

# plot variance
plt.figure()
plt.imshow(bproj(var_Vc), vmin=-0.04, vmax=0.04)
plt.colorbar()
plt.title('variance profile')



#%% plot component
c = 3

plt.figure()
plt.imshow(U[c,:,:], vmin=-0.001, vmax=0.001)
plt.colorbar()
plt.title('component spatial profile')

plt.figure()
plt.plot(Vc[:,c])
plt.title('component trace')






#%% MOU model

from mou_model import MOU

comp_est = np.arange(20) # components for estimation
# comp_est = np.delete(comp_est,[1]) # remove supposed artifact (component 1)
comp_est = np.delete(comp_est,[0,1]) # remove global component and supposed artifact (component 1)
Vc_est = Vc[:,comp_est]

mou_est = MOU() # create a MOU model to fit the data
mou_est.fit(Vc_est, i_tau_opt=2)

# plot estimated EC
plt.figure()
plt.plot(Vc_est)
plt.title('time courses')

# plot estimated EC
plt.figure()
plt.imshow(mou_est.J, vmin=0)
plt.title('estimated EC')

# plot estimated Sigma (variance of fluctuating inputs)
plt.figure()
plt.imshow(mou_est.Sigma, vmin=0)
plt.title('estimated Sigma')
