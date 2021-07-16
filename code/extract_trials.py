from load_task_data_as_pandas_df import extract_session_data_and_save, fix_dates
import numpy as np
import h5py
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path


def extract_frames(cond, sf, root_paths):
    file_path = root_paths / "SVD_data/Vc.mat"
    f = h5py.File(file_path, 'r')
    fc = np.array(f['frameCnt'])
    stimulus_start_frame_ids = np.cumsum(fc[:, 1]) 
    cues = list()
    for ids in range(fc.shape[0]):
        relative_frame_ids_of_individual_cues = (cond[ids] * np.linspace(0, (sf / 2.) * 5, 6))[(cond[ids]) != 0]
        if np.sum(relative_frame_ids_of_individual_cues) != 0:
            a = np.ceil(stimulus_start_frame_ids[ids] + relative_frame_ids_of_individual_cues)
            cues.append(np.arange(a[0], a[-1]+ 8))
    b = np.array(cues, dtype=object)
    b = b[np.array([y.size > 0 for y in b])]
    return b


def trial_extraction(data, what_cond, sf, root_paths):
    if isinstance(what_cond[0], (np.ndarray)) == True:
        trials = list()
        b = extract_frames(what_cond, sf, root_paths)
        for frame in b:
            #### extract trials in frames
            innt_frame = frame.astype('int')
            if innt_frame[-1] < data.shape[0]:
                trials.append(data[innt_frame, :])

    return np.array(trials, dtype=object)


def extract_all_trials(cond_list, sf, root_paths, mouse_ids, run_extraction, save=False):
    ##### extract trials for all condition in a list
    if run_extraction:
        sessions = extract_session_data_and_save(root_paths=root_paths, mouse_ids=mouse_ids, reextract=True)
    else:
        # load data
        sessions = pkl.load(open(root_paths[0] / "task_data/MS_task_V2_1_extracted_data.pkl", 'rb'))
        sessions = fix_dates(sessions)

    file_path = root_paths[0] / 'SVD_data/Vc.mat'
    f = h5py.File(file_path, 'r')
    Vc = np.array(f['Vc'])

    end = dict.fromkeys(cond_list, 0)
    for cond in cond_list:
        end[cond] = trial_extraction(Vc, sessions[cond], sf, root_paths[0])

    if save:
        save_trials(root_paths[0] / "task_data/", end)

    return end


############

def first_frams(trials, lim):
    ###### cut trials into sepcific frame lenghts
    z = list()
    for x in trials:
        if x.shape[0] >= lim:
            z.append(x[:lim, :])
    return np.array(z)


############
def save_trials(path, trials):
    pkl.dump(trials, open(path / 'MS_task_V2_1_extracted_trials.pkl', 'wb'))


def load_trials(path):
    trials = pkl.load(open(path / "task_data/MS_task_V2_1_extracted_trials.pkl", 'rb'))
    return trials


########################################################################################################
if __name__ == '__main__':
    mouse_ids = ['GN06']
    root_paths = [ Path(__file__).parent.parent / Path('data/GN06/2021-01-20_10-15-16') ]
    conds = ["cues_right_vis", "cues_left_vis", "cues_left_tact", "cues_right_tact"]
    extract_trials = True
    samplingfreq = 15

    if extract_trials:
        trials = extract_all_trials(conds, samplingfreq, root_paths, mouse_ids, run_extraction=True, save=True) #Extraction 1.True
    else:
        trials = load_trials(root_paths[0])

    frames = 45
    plt.figure()
    plt.plot(first_frams(trials["cues_right_vis"], frames).mean(0))
    plt.xlabel('frame')
    plt.ylabel('mean')
    plt.title('cues_right_vis')
