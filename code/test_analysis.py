import h5py
import numpy
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from load_task_data_as_pandas_df import extract_session_data_and_save

from data import DecompData
from measurements import mean
from matplotlib import pyplot as plt

import itertools

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
svd = DecompData( sessions, np.array(f["Vc"]), np.array(f["U"]), np.array(trial_starts) )

trial_preselection = ((svd.n_targets == 6) & (svd.n_distractors == 0) &
                      (svd.auto_reward == 0) & (svd.both_spouts == 1))
svd_pre = svd[ :, trial_preselection ]


#extract conds
modal_keys = ['visual', 'tactile', 'vistact']
modal_range = range(3)

side_keys = ['right', 'left']
side_range = range(2)


'''
svd_cond = [svd_pre[:,(svd_pre.modality == modal) & (svd_pre.target_side_left == side)] for modal, side in itertools.product(modal_range,side_range)]
cond_keys =  list(itertools.product(modal_keys,side_keys))
cond_keys_str = [f"{s}_{m}" for m, s in cond_keys]
'''

svd_pre.conditions = ([(svd_pre.modality == modal) & (svd_pre.target_side_left == side) for modal, side in itertools.product(modal_range,side_range)])



print("Class", len(svd_pre._conditions))
print(svd_pre.conditions)




#cond_mean = mean(svd_cond[30:75,:]) #mean of stimulusframes

