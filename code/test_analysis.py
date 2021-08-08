import h5py
import numpy
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from matplotlib import pyplot as plt
import itertools


from data import DecompData

##Need better solution
import sys
sys.path.append(Path(__file__).parent)


from features import measurements
from loading import load_task_data_as_pandas_df

plt_mean = False

data_path = Path(__file__).parent.parent / Path('data')
if not (data_path/'extracted_data.pkl').exists() :
    # load behavior data
    sessions = load_task_data_as_pandas_df.extract_session_data_and_save(root_paths=[data_path], mouse_ids=["GN06"], reextract=False)
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
svd = DecompData( sessions, np.array(f["Vc"]), np.array(f["U"]), np.array(trial_starts) )


#define different conds
modal_keys = ['visual', 'tactile', 'vistact']
modal_range = range(3)

side_keys = ['right', 'left']
side_range = range(2)

#filter for all conds
trial_preselection = ((svd.n_targets == 6) & (svd.n_distractors == 0) & (svd.auto_reward == 0) & (svd.both_spouts == 1))

#set condition filter
svd.conditions = ([(svd.modality == modal) & (svd.target_side_left == side) & trial_preselection for modal, side in itertools.product(modal_range,side_range)])

print(svd.conditions)
#print(svd.conditions[:,:,:])




cond_mean = measurements.mean(svd.conditions[0][30:75,:]) #mean of stimulusframes for first cond

