import matplotlib.pyplot as plt
import h5py
from glob import glob
import numpy as np


date_folder = r'W:\Multisensory_task\MS_task_V2_2\GN10\2021-03-22_09-29-23'
file_paths = glob(date_folder + r'\task_data\*.h5')
file_paths = np.sort(file_paths)

for trial_file in file_paths:
    try:
        h5f = h5py.File(trial_file, 'r')

        print('modality: %d, target_side: %d' % (int(h5f['modality'][0]), int(h5f['target_side_left'][0])))

        #
        data = np.array(h5f['DI'])
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(data, aspect='auto')
        ax[0].set_xlabel('Samples @ 10kHz')
        ax[0].set_yticks(range(8), ['0: lick_left', '1: lick_right', '2: camera_trigger', '3: task_feedback_trigger', '4: airpuff_left', '5: airpuff_right', '6: water_valve_left', '7: water_valve_right'])
        ax[0].set_title('DI')

        #

        speed = np.diff(np.array(h5f['wheel'])) / np.array(h5f['n_DI_samples_since_last_wheel_update'][:-1])
        ax[1].imshow(np.array([(h5f['photodiode'][:-1] - np.array(h5f['photodiode']).min()) / (np.array(h5f['photodiode']).max() - np.array(h5f['photodiode']).min()),
                             (h5f['wheel'][:-1] - np.array(h5f['wheel']).min()) / (np.array(h5f['wheel']).max() - np.array(h5f['wheel']).min()),
                             (speed - speed.min()) / (speed.max() - speed.min())]).reshape(3, -1), aspect='auto')
        ax[1].set_yticks(range(3), ['photodiode', 'wheel', 'speed'])
        ax[1].set_xlabel('Samples but irregular')

        plt.show()
    except Exception as e:
        pass
    #
#
