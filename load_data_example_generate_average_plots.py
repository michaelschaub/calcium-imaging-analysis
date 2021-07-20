"""
Loading Widefield-Imaging data:
After Compression, data is stored in the Folder SVD. File 'Vc.mat' usually contrians everything necessary.
"""

import h5py
import numpy as np
from load_task_data_as_pandas_df import extract_session_data_and_save
import matplotlib.pyplot as plt


# Load Session Data
mouse_ids = ['GN06']
root_paths = [r'']
run_extraction = True

sessions = extract_session_data_and_save(root_paths=root_paths, mouse_ids=mouse_ids, reextract=False)
#

np[:, np.nan]
np[np.nan, :]

# Load imaging data
# file_path = r'E:\Uni\Sciebo\ERS_calcium_analysis\Example_data\GN06\2021-01-20_10-15-16\SVD_data\Vc.mat'
file_path = r'GN06\2021-01-20_10-15-16\SVD_data\Vc.mat'
arrays = {}
f = h5py.File(file_path, 'r')
frameCnt = np.array(f['frameCnt'])

U = np.array(f['U'])
U_shape = U.shape
# reshape for the dot product
U = U.reshape([U.shape[0], -1])

#
fig, ax = plt.subplots(2, 3)
for modality_id in range(3):
    for target_side in range(2):
        print(modality_id, target_side)

        trial_starts = np.cumsum(frameCnt[:, 1])

        # Ensure that both have the same length
        if len(trial_starts) > sessions.shape[0]:
            trial_starts = trial_starts[:sessions.shape[0]]
        if len(trial_starts) < sessions.shape[0]:
            sessions = sessions[:len(trial_starts)]
        #

        # get left tactile trials
        selected_trials = ((sessions.modality == modality_id) & (sessions.target_side_left == target_side) & (sessions.n_targets == 6) & (sessions.n_distractors == 0))
        n_trials = np.sum(selected_trials)
        print('n_trials: ', n_trials)
        # frame_ids = trial_starts[left_tactile_trials]
        # np.repeat(np.reshape(np.arange(30, 75), (1, 45)), 10, axis=0)

        trial_frame_range = np.arange(30, 75)
        selected_frame_ids = np.repeat(np.reshape(trial_frame_range, (1, len(trial_frame_range))), n_trials, axis=0) + \
            np.repeat(np.reshape(trial_starts[selected_trials], (n_trials, 1)), len(trial_frame_range), axis=1)

        trial_frame_range = np.arange(15, 30)
        baseline_frame_ids = np.repeat(np.reshape(trial_frame_range, (1, len(trial_frame_range))), n_trials, axis=0) + \
            np.repeat(np.reshape(trial_starts[selected_trials], (n_trials, 1)), len(trial_frame_range), axis=1)

        # np.repeat(np.reshape(trial_frame_range, (1, len(trial_frame_range))), n_trials, axis=0)

        # currently there is a 2sec (@15Hz -> 30frames) baseline before the stimulus. This information is included in the task_data
        baseline_frames = 30
        n_frames = 30

        # either load entire file or selected frames
        # frame_ids = range(150 + baseline_frames, 150 + baseline_frames + n_frames)
        plt_mean = True
        if plt_mean:
            # selected_frame_ids = selected_frame_ids.flatten()
            # baseline_frame_ids = baseline_frame_ids.flatten()

            Vc = np.array(f['Vc'][selected_frame_ids.flatten(), :])
            Vc_baseline = np.array(f['Vc'][baseline_frame_ids.flatten(), :])

            # average over all frames, if loaded accordingly
            Vc_mean = Vc.reshape([-1, U_shape[0]]).mean(axis=0)
            Vc_baseline_mean = Vc_baseline.reshape([-1, U_shape[0]]).mean(axis=0)

            """
            Visualization:
            """

            # compute dot product and reshape back into 2D frame
            average_frame = np.dot(Vc_mean - Vc_baseline_mean, U).reshape([U_shape[1], U_shape[2]])

            # plot
            # plt.imshow(average_frame, vmin=-0.04, vmax=0.04)
            # plt.imshow(average_frame, vmin=-0.04, vmax=0.04, cmap=plt.hot())
            if modality_id == 1:
                im = ax[target_side, modality_id].imshow(average_frame, vmin=-0.01, vmax=0.01)
            else:
                im = ax[target_side, modality_id].imshow(average_frame, vmin=-0.04, vmax=0.04)
            #
        else:
            # # define the frames you want to average over
            # frame_ids = range(150 + 30, 150 + 60)
            #
            # # average over frames and compute component weights
            # Vc_mean = Vc[frame_ids, :].reshape([-1, U_shape[0]]).mean(axis=0)

            # this plots z-score now
            Vc = np.array(f['Vc'][selected_frame_ids.flatten(), :])
            Vc_baseline = np.array(f['Vc'][baseline_frame_ids.flatten(), :])

            Vc = Vc.reshape([selected_frame_ids.shape[0], selected_frame_ids.shape[1], -1])
            Vc_mean = Vc.mean(axis=1)

            Vc_baseline = Vc_baseline.reshape([baseline_frame_ids.shape[0], baseline_frame_ids.shape[1], -1])
            Vc_baseline_mean = Vc_baseline.mean(axis=1)

            """
            Visualization:
            """
            # compute dot product and reshape back into 2D frame
            average_stimulus_frames = np.dot(Vc_mean, U)
            average_baseline_frames = np.dot(Vc_baseline_mean, U)

            average_stimulus_frames = average_stimulus_frames - average_baseline_frames
            z_score = ((average_stimulus_frames.mean(axis=0)) / (average_stimulus_frames.std(axis=0))).reshape([U_shape[1], U_shape[2]])

            # plot
            # plt.imshow(average_frame, vmin=-0.04, vmax=0.04)
            # plt.imshow(average_frame, vmin=-0.04, vmax=0.04, cmap=plt.hot())
            if modality_id == 1:
                im = ax[target_side, modality_id].imshow(z_score, vmin=-1.5, vmax=1.5)
            else:
                im = ax[target_side, modality_id].imshow(z_score, vmin=-3, vmax=3)
            #
        #

        # ax[target_side, modality_id].colorbar()
        fig.colorbar(im, ax=ax[target_side, modality_id])
        plt.draw()
        plt.pause(1)
    #
#
plt.show()
