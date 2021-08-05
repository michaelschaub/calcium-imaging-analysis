import h5py
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from load_task_data_as_pandas_df import extract_session_data_and_save
from data import SVDData
from matplotlib import pyplot as plt

plt_mean = False

data_path = Path(__file__).parent.parent / Path('data')
if not (data_path/'extracted_data.pkl').exists() :
    # load behavior data
    sessions = extract_session_data_and_save(root_paths=[data_path], mouse_ids=["GN06"], reextract=False)
    with open( data_path / 'extracted_data.pkl', 'wb') as handle:
        pkl.dump(sessions, handle)
else:
    # load saved data
    with open( data_path / 'extracted_data.pkl', 'rb') as handle:
        sessions = pkl.load(handle)
    print("Loaded pickled data.")

file_path = data_path / "GN06" / Path('2021-01-20_10-15-16/SVD_data/Vc.mat')
f = h5py.File(file_path, 'r')

frameCnt = np.array(f['frameCnt'])
trial_starts = np.cumsum(frameCnt[:, 1])
svd = SVDData( sessions, np.array(f["Vc"]), np.array(f["U"]), np.array(trial_starts) )

trial_preselection = ((svd.n_targets == 6) & (svd.n_distractors == 0) &
                      (svd.auto_reward == 0) & (svd.both_spouts == 1))
svd_pre = svd[ :, trial_preselection ]

modality_keys = ['visual', 'tactile', 'vistact']
target_side_keys = ['right', 'left']
fig, ax = plt.subplots(2, 3)

for modality_id in range(3):
    for target_side in range(2):
        # get the trials to use
        selected_trials = ((svd_pre.modality == modality_id) & (svd_pre.target_side_left == target_side))

        # stimulus frames
        selected_frames = svd_pre[ 30:75, selected_trials ]

        # baseline frames (1sec before stimulus)
        baseline_frames = svd_pre[ 15:30, selected_trials ]

        # calculate either the mean over trials or the z_score
        if plt_mean:
            # average over all frames, if loaded accordingly
            Vc_mean = selected_frames.temporals_flat.mean(axis=0)
            Vc_baseline_mean = baseline_frames.temporals_flat.mean(axis=0)

            """
            Visualization:
            """
            # compute dot product and reshape back into 2D frame
            average_frame = np.einsum( "n,nij->ij", Vc_mean-Vc_baseline_mean, selected_frames.spatials )

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
#
plt.show()
