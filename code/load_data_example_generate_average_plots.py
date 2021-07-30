"""
Loading Widefield-Imaging data:
After Compression, data is stored in the Folder SVD. File 'Vc.mat' usually contains everything necessary.
"""

import h5py
import numpy as np
from load_task_data_as_pandas_df import extract_session_data_and_save
import matplotlib.pyplot as plt
from pathlib import Path
import pickle as pkl


# calculate either the mean over trials or the z_score
plt_mean = False

# Load Session Data
mouse_ids = ['GN06']
root_paths = [ Path(__file__).parent.parent / Path('data') ]
run_extraction = True

if run_extraction:
    # load behavior data
    sessions = extract_session_data_and_save(root_paths=root_paths, mouse_ids=mouse_ids, reextract=False)
    with open( root_paths[0] / 'extracted_data.pkl', 'wb') as handle:
        pkl.dump(sessions, handle)
else:
    # load saved data
    with open( root_paths[0] / 'extracted_data.pkl', 'rb') as handle:
        sessions = pkl.load(handle)

# pre-selection moved after length match

# # look at all difficulties
# trial_preselection = ((sessions.auto_reward == 0) & (sessions.both_spouts == 1))
#

# Load imaging data
file_path = root_paths[0] / mouse_ids[0] / Path('2021-01-20_10-15-16/SVD_data/Vc.mat')
f = h5py.File(file_path, 'r')
frameCnt = np.array(f['frameCnt'])

# by taking the cumsum we get the frame_ids when thee individual trials started. This works, because the first element
# is the length of the pre-session baseline (trial_id -1, 150 frames)
trial_starts = np.cumsum(frameCnt[:, 1])

# Ensure that both have the same length
if len(trial_starts) > sessions.shape[0]:
    trial_starts = trial_starts[:sessions.shape[0]]
if len(trial_starts) < sessions.shape[0]:
    sessions = sessions[:len(trial_starts)]
#
# pre-select trials to look at
# if looking only at the simplest cases (detection 6 vs. 0)
trial_preselection = ((sessions.n_targets == 6) & (sessions.n_distractors == 0) &
                      (sessions.auto_reward == 0) & (sessions.both_spouts == 1))


U = np.array(f['U'])
U_shape = U.shape  # get the shape of the frames before reshaping for dot-product later
# reshape for the dot product
U = U.reshape([U.shape[0], -1])  # now flatten x, y pixel dimensions

# loop over the modality and target_side conditiions
modality_keys = ['visual', 'tactile', 'vistact']
target_side_keys = ['right', 'left']
fig, ax = plt.subplots(2, 3)
for modality_id in range(3):
    for target_side in range(2):
        # get the trials to use
        selected_trials = (trial_preselection & (sessions.modality == modality_id) & (sessions.target_side_left == target_side))

        # get the frame_ids of stimulus frames in the selected trials
        trial_frame_range = np.arange(30, 75)  # stimulus frames
        # CHANGED: excluded last trial, since Vc ends at its start (is it supposed to?)
        selected_frame_ids = np.array( trial_frame_range[np.newaxis, :] + trial_starts[:-1][selected_trials[:-1], np.newaxis], dtype=int)

        # get the baseline_ids to estimate a general baseline
        trial_frame_range = np.arange(15, 30)  # baseline frames (1sec before stimulus)
        # I'm using the trial_preselection just in case there are some weird trials in the beginning or end of the
        # session, where the animal might be still impatient or already disengaged
        # CHANGED: excluded last trial, since Vc ends at its start (is it supposed to?)
        baseline_frame_ids = np.array( trial_frame_range[np.newaxis, :] + trial_starts[:-1][trial_preselection[:-1], np.newaxis], dtype=int)

        # calculate either the mean over trials or the z_score
        if plt_mean:
            Vc = np.array(f['Vc'][selected_frame_ids.flatten(), :])
            Vc_baseline = np.array(f['Vc'][baseline_frame_ids.flatten(), :])

            # average over all frames, if loaded accordingly
            Vc_mean = Vc.mean(axis=0)
            Vc_baseline_mean = Vc_baseline.mean(axis=0)

            """
            Visualization:
            """
            # compute dot product and reshape back into 2D frame
            average_frame = np.dot(Vc_mean - Vc_baseline_mean, U).reshape([U_shape[1], U_shape[2]])

            # plot
            im = ax[target_side, modality_id].imshow(average_frame, vmin=-0.02, vmax=0.02)
        else:
            # this plots z-score now
            Vc = np.array(f['Vc'][selected_frame_ids.flatten(), :])
            Vc_baseline = np.array(f['Vc'][baseline_frame_ids.flatten(), :])

            Vc = Vc.reshape([selected_frame_ids.shape[0], selected_frame_ids.shape[1], -1])
            Vc_mean = Vc.mean(axis=1)

            Vc_baseline = Vc_baseline.reshape([baseline_frame_ids.shape[0], baseline_frame_ids.shape[1], -1])
            Vc_baseline_mean = Vc_baseline.mean(axis=1).mean(axis=0)

            """
            Visualization:
            """
            # compute dot product and reshape back into 2D frame
            average_stimulus_frames = np.dot(Vc_mean, U)
            average_baseline_frames = np.dot(Vc_baseline_mean, U)

            average_stimulus_frames = average_stimulus_frames - average_baseline_frames
            z_score = ((average_stimulus_frames.mean(axis=0)) / (average_stimulus_frames.std(axis=0))).reshape([U_shape[1], U_shape[2]])

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
