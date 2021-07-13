"""
Loading Widefield-Imaging data:
After Compression, data is stored in the Folder SVD. File 'Vc.mat' usually contrians everything necessary.
"""

import h5py
import numpy as np
from pathlib import Path


file_path = [ Path(__file__).parent.parent / Path('data/GN06/2021-01-20_10-15-16/SVD_data\Vc.mat') ]
arrays = {}
f = h5py.File(file_path, 'r')
if False:
    pass
    # load full file
    # for k, v in f.items():
    #     arrays[k] = np.array(v)
    #
else:
    frameCnt = np.array(f['frameCnt'])

    # currently there is a 2sec (@15Hz -> 30frames) baseline before the stimulus. This information is included in the task_data
    baseline_frames = 30
    n_frames = 30

    # either load entire file or selected frames
    frame_ids = np.arange(150 + baseline_frames, 150 + baseline_frames + n_frames)
    Vc = np.array(f['Vc'][frame_ids, :])
    U = np.array(f['U'])
#


"""
Visualization:
"""

import matplotlib.pyplot as plt

# keep the shape for later
U_shape = U.shape

# reshape for the dot product
U = U.reshape([U.shape[0], -1])

if True:
    # average over all frames, if loaded accordingly
    Vc_mean = Vc.reshape([-1, U_shape[0]]).mean(axis=0)
else:
    # define the frames you want to average over
    frame_ids = range(150 + 30, 150 + 60)

    # average over frames and compute component weights
    Vc_mean = Vc[frame_ids, :].reshape([-1, U_shape[0]]).mean(axis=0)
#

# compute dot product and reshape back into 2D frame
average_frame = np.dot(Vc_mean, U).reshape([U_shape[1], U_shape[2]])

# plot
plt.imshow(average_frame, vmin=-0.04, vmax=0.04)

plt.show()
