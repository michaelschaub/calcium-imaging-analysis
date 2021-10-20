import numpy as np
import h5py
from pathlib import Path
import warnings
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from loading import load_task_data_as_pandas_df #import extract_session_data_and_save
from data import DecompData

files_Vc = snakemake.input["Vc"]
task_files =  snakemake.input["tasks"]

### should definitly be reworked ###

task_files_split = task_files[0].split("/")
data_path = "/".join(task_files_split[:-3])
mouse_id = task_files_split[-3]
sessions = load_task_data_as_pandas_df.extract_session_data_and_save(
        root_paths=[Path(data_path)], mouse_ids=[mouse_id], reextract=False)
###   ###

if len(files_Vc) > 1:
    warnings.warn("Combining different dates may still be buggy!")

trial_starts = []
Vc = []
U = []
start = 0
for file_Vc in files_Vc:
    f = h5py.File(file_Vc, 'r')
    frameCnt = np.array(f['frameCnt'])
    Vc.append(np.array(f["Vc"]))
    U.append(np.array(f["U"]))
    assert (U[-1] == U[0]).all(), "Combining different dates with different Compositions is not yet supported"
    trial_starts.append(np.cumsum(frameCnt[:-1, 1]) + start)
    start += Vc[-1].shape[0]

trial_starts = np.concatenate( trial_starts )
Vc = np.concatenate( Vc )
U = U[0]

svd = DecompData( sessions, Vc, U, trial_starts )
svd.save( snakemake.output[0] )
