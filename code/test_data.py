import h5py
import numpy as np
import pandas as pd
import pickle as pkl
#from pathlib import Path
import pathlib
import math

###Too complicated
import sys
sys.path.append(pathlib.Path(__file__).parent)
'''
folders = ['features','loading']
file_paths =  [Path(__file__).parent / Path(folder) for folder in folders]
for  f in file_paths:
    print(f)
    sys.path.append(f)
'''
###

from loading import load_task_data_as_pandas_df #import extract_session_data_and_save
from data import DecompData
from features import Means, Raws
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

plt_mode = "raw" # should be from ["mean", "z_score", "raw"]
raw_course_graining = 5

force_extraction = False

data_path = pathlib.Path(__file__).parent.parent/'data'
svd_path = data_path/'output'/'GN06'/'svd_data.h5'
if (not svd_path.exists()) or force_extraction:
    if (not (data_path/'input/extracted_data.pkl').exists()) or force_extraction:
        # load behavior data
        sessions = load_task_data_as_pandas_df.extract_session_data_and_save(root_paths=[data_path/"input"], mouse_ids=["GN06"], reextract=False)
        with open( data_path/'input/extracted_data.pkl', 'wb') as handle:
            pkl.dump(sessions, handle)
    else:
        # load saved data
        with open( data_path/'input/extracted_data.pkl', 'rb') as handle:
            sessions = pkl.load(handle)
        print("Loaded pickled data.")

    file_path = data_path/'input'/'GN06'/'2021-01-20_10-15-16'/'SVD_data'/'Vc.mat'
    f = h5py.File(file_path, 'r')

    frameCnt = np.array(f['frameCnt'])
    trial_starts = np.cumsum(frameCnt[:-1, 1])
    svd = DecompData( sessions, np.array(f["Vc"]), np.array(f["U"]), np.array(trial_starts) )
    svd.save(str(svd_path))
else:
    svd = DecompData.load(str(svd_path))
    print("Loaded DecompData object.")


trial_preselection = ((svd.n_targets == 6) & (svd.n_distractors == 0) &
                      (svd.auto_reward == 0) & (svd.both_spouts == 1))

print(trial_preselection.shape)
print(svd[:,:].shape)
svd_pre = svd[ trial_preselection ]

modality_keys = ['visual', 'tactile', 'vistact']
target_side_keys = ['right', 'left']

if plt_mode in ["mean", "z_score"]:

    fig, ax = plt.subplots(2, 3)

    for modality_id in range(3):
        for target_side in range(2):
            # get the trials to use
            selected_trials = ((svd_pre.modality == modality_id) & (svd_pre.target_side_left == target_side))

            # stimulus frames
            selected_frames = svd_pre[ selected_trials, 30:75 ]

            # baseline frames (1sec before stimulus)
            baseline_frames = svd_pre[ selected_trials, 15:30 ]

            # calculate either the mean over trials or the z_score
            if plt_mode == "mean":
                # average over all frames, if loaded accordingly
                #Vc_mean = selected_frames.temporals_flat.mean(axis=0)
                #Vc_baseline_mean = baseline_frames.temporals_flat.mean(axis=0)
                frames_corrected = Means(selected_frames - Means( baseline_frames ))

                """
                Visualization:
                """
                # compute dot product and reshape back into 2D frame
                #average_frame = np.einsum( "n,nij->ij", Vc_mean-Vc_baseline_mean, selected_frames.spatials )
                #average_frame = np.mean( frames_corrected.pixel[:], 0)
                #average_frame = frames_corrected.pixel[:,:,:]
                #print(average_frame.shape)
                print("averages")
                average_frame = frames_corrected.mean.pixel[0,:,:]
                print(average_frame.shape)

                # plot
                im = ax[target_side, modality_id].imshow(average_frame, vmin=-0.02, vmax=0.02)
            else:
                # this plots z-score now
                Vc = selected_frames.temporals
                Vc_mean = Vc.mean(axis=1)

                Vc_baseline = baseline_frames.temporals
                Vc_baseline_mean = Vc_baseline.mean(axis=1).mean(axis=0)

                """
                Visualization:
                """
                # compute dot product and reshape back into 2D frame
                U = selected_frames.spatials
                average_stimulus_frames = np.tensordot(Vc_mean, U, (-1,0) )
                average_baseline_frames = np.tensordot(Vc_baseline_mean, U, (-1,0) )

                average_stimulus_frames = average_stimulus_frames - average_baseline_frames
                z_score = ((average_stimulus_frames.mean(axis=0)) / (average_stimulus_frames.std(axis=0)))

                # plot
                im = ax[target_side, modality_id].imshow(z_score, vmin=-5, vmax=5)
            #

            fig.colorbar(im, ax=ax[target_side, modality_id])
            ax[target_side, modality_id].set_title(modality_keys[modality_id] + ' - ' + target_side_keys[target_side])
            ax[target_side, modality_id].set_xticks([])
            ax[target_side, modality_id].set_yticks([])
            plt.draw()
            plt.pause(0.1)
    #
    plt.show()
elif plt_mode == "raw":
    for modality_id in range(3):
        for target_side in range(2):
            # get the trials to use
            selected_trials = ((svd_pre.modality == modality_id) & (svd_pre.target_side_left == target_side))

            # stimulus frames
            selected_frames = svd_pre[ selected_trials, 30:75 ]

            # baseline frames (1sec before stimulus)
            baseline_frames = svd_pre[ selected_trials, 15:30 ]

            frames_corrected = Raws(selected_frames - Means( baseline_frames ))
            average_frames = frames_corrected.mean.pixel[:,:,:]
            print(average_frames.shape)

            """
            Visualization:
            """
            fig, ax = plt.subplots()

            cg = raw_course_graining

            def draw_frame(t):
                im = ax.imshow(
                        np.mean( average_frames[cg*t:min(cg*(t+1),average_frames.shape[0]), :, :], axis=0),
                        vmin=-0.02, vmax=0.02)
                print(average_frames[cg*t:min(cg*(t+1),average_frames.shape[0]), :, :].shape)
                #fig.colorbar(im, ax=ax[target_side, modality_id])
            ani = FuncAnimation( fig, draw_frame, frames=math.ceil(average_frames.shape[0]/cg), interval=100, repeat=True)
            plt.show()

else:
    raise ValueError("plt_mode not known")
