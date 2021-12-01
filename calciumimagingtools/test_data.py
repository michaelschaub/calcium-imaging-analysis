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

plt_mode = "raw_z_score" # should be from ["mean", "z_score", "raw", "raw_z_score", None]
plt_mode = "raw"
raw_course_graining = 1
animation_slowdown = 1

save_feat = True
load_feat = True

force_extraction = False


resc_path = pathlib.Path(__file__).parent.parent/'resources'
resl_path = pathlib.Path(__file__).parent.parent/'results'
svd_path = resl_path/'GN06/SVD/data.h5'
if (not svd_path.exists()) or force_extraction:
    if (not (resc_path/'extracted_data.pkl').exists()) or force_extraction:
        # load behavior data
        sessions = load_task_data_as_pandas_df.extract_session_data_and_save(root_paths=[resc_path/"experiments"], mouse_ids=["GN06"], reextract=False)
        with open( resc_path/'extracted_data.pkl', 'wb') as handle:
            pkl.dump(sessions, handle)
    else:
        # load saved data
        with open( resc_path/'extracted_data.pkl', 'rb') as handle:
            sessions = pkl.load(handle)
        print("Loaded pickled data.")

    file_path = resc_path/'experiments'/'GN06'/'2021-01-20_10-15-16'/'SVD_data'/'Vc.mat'
    f = h5py.File(file_path, 'r')

    frameCnt = np.array(f['frameCnt'])
    trial_starts = np.cumsum(frameCnt[:-1, 1])
    svd = DecompData( sessions, np.array(f["Vc"]), np.array(f["U"]), np.array(trial_starts) )
    svd.save(str(svd_path))
else:
    svd = DecompData.load(svd_path)
    print(f"Loaded DecompData object from '{str(svd_path)}'.")

print(f"SVD is saved at {svd.savefile}")

trial_preselection = ((svd.n_targets == 6) & (svd.n_distractors == 0) &
                      (svd.auto_reward == 0) & (svd.both_spouts == 1))

print(trial_preselection.shape)
print(svd[:,:].shape)
svd_pre = svd[ trial_preselection ]

modality_keys = ['visual', 'tactile', 'vistact']
target_side_keys = ['right', 'left']

save_files = [ [ resl_path/f"{plt_mode}_{mod}_{side}.h5" for side in target_side_keys ] for mod in modality_keys ]
data_save_file = resl_path/f"data.h5"

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

                if load_feat and save_files[modality_id][target_side].exists():
                    frames_corrected = Means.load(save_files[modality_id][target_side])
                    print(f"Loaded {frames_corrected._savefile}")
                else:
                    # average over all frames, if loaded accordingly
                    #Vc_mean = selected_frames.temporals_flat.mean(axis=0)
                    #Vc_baseline_mean = baseline_frames.temporals_flat.mean(axis=0)
                    frames_corrected = Means.create(selected_frames - Means.create( baseline_frames ))

                    if save_feat:
                        frames_corrected.save(save_files[modality_id][target_side], data_file=data_save_file)
                        print(f"Saved into {frames_corrected._savefile}")

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

                # plot
                im = ax[target_side, modality_id].imshow(average_frame, vmin=-0.02, vmax=0.02)
            else:
                # this plots z-score now
                Vc = selected_frames.temporals
                Vc_mean = Vc.mean(axis=1)

                Vc_baseline = baseline_frames.temporals
                Vc_baseline_mean = Vc_baseline.mean(axis=1)#.mean(axis=0)

                """
                Visualization:
                """
                # compute dot product and reshape back into 2D frame
                U = selected_frames.spatials
                average_stimulus_frames = np.tensordot(Vc_mean, U, (-1,0) )
                average_baseline_frames = np.tensordot(Vc_baseline_mean, U, (-1,0) )

                average_stimulus_frames = average_stimulus_frames - average_baseline_frames.mean(axis=0)
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
elif plt_mode in ["raw", "raw_z_score" ]:
    for modality_id in range(3):
        for target_side in range(2):
            print(f"{modality_keys[modality_id]}, {target_side_keys[target_side]}")
            # get the trials to use
            selected_trials = ((svd_pre.modality == modality_id) & (svd_pre.target_side_left == target_side))

            # stimulus frames
            selected_frames = svd_pre[ selected_trials, 30:75 ]

            # baseline frames (1sec before stimulus)
            baseline_frames = svd_pre[ selected_trials, 15:30 ]

            if plt_mode == "raw":
                if load_feat and save_files[modality_id][target_side].exists():
                    frames_corrected = Raws.load(save_files[modality_id][target_side])
                    print(f"Loaded {frames_corrected._savefile}")
                else:
                    #frames_corrected = Raws.create(selected_frames)
                    frames_corrected = Raws.create(selected_frames - Means.create( baseline_frames ))

                    if save_feat:
                        frames_corrected.save(save_files[modality_id][target_side], data_file=data_save_file)
                        print(f"Saved into {frames_corrected._savefile}")

                plot_frames = frames_corrected.mean.pixel[:,:,:]
            else:
                frames_corrected = Raws.create(selected_frames - Means.create( baseline_frames ))
                average_frames = frames_corrected.mean.pixel[:,:,:]
                baseline_frames = Means.create( baseline_frames ).pixel[:,:,:]
                plot_frames = average_frames / baseline_frames.std()

            """
            Visualization:
            """
            fig, ax = plt.subplots()

            cg = raw_course_graining
            im = [None] # the image has to be containered to pass it as reference

            inter = animation_slowdown * 1e3*cg/15

            def draw_frame(t):
                if im[0] is None:
                    im[0] = ax.imshow(
                            np.mean( plot_frames[cg*t:min(cg*(t+1),plot_frames.shape[0]), :, :], axis=0),
                            vmin=-0.02, vmax=0.02)
                    cb = fig.colorbar(im[0], ax=ax)
                else:
                    im[0].set_data(np.mean( plot_frames[cg*t:min(cg*(t+1),plot_frames.shape[0]), :, :], axis=0))
            ani = FuncAnimation( fig, draw_frame,
                                frames=math.ceil(plot_frames.shape[0]/cg), interval=inter, repeat=True)
            plt.show()

elif plt_mode is None:
    pass
else:
    raise ValueError("plt_mode not known")
