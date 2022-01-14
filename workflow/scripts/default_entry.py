import numpy as np
import h5py
import warnings

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.loading import load_task_data_as_pandas_df, alignment #import extract_session_data_and_save
from ci_lib import DecompData

# redirect std_out to log file
snakemake_tools.redirect_to_log(snakemake)
snakemake_tools.save_conf(snakemake, sections=["entry","parcellation"]) #fixed a bug as we dont apply parcellation to SVD and then prefilter fails to compare config as it won't contain parcellation
timer_start = snakemake_tools.start_timer()




files_Vc = snakemake.input["Vc"]
task_files = snakemake.params["task_structured"] #snakemake.input["tasks"]
trans_paths = snakemake.input["trans_params"]


mouse_dates_str = snakemake.params["mouse_dates_str"]

### should definitly be reworked ###

#task_files_split = task_files[0].split("/")
#print("task_files_split",task_files_split)
#data_path = "/".join(task_files_split[:-3])
#print("entry data path",data_path)
#mouse_id = task_files_split[-3]
#dates

sessions = load_task_data_as_pandas_df.extract_session_data_and_save(
        root_paths=task_files, mouse_dates_str = mouse_dates_str ,reextract=True) #reextraction needs to be done for different set of dates otherwise session will have wrong dims
print("Loaded task data")
###   ###

if len(files_Vc) > 1:
    warnings.warn("Combining different dates may still be buggy!")

trial_starts = []
Vc = []
U = []
start = 0
for file_Vc,trans_path in zip(files_Vc,trans_paths):
    f = h5py.File(file_Vc, 'r')

    #Aligns spatials for each date with respective trans_params
    alignend_U = alignment.align_spatials_path(np.array(f["U"]),trans_path)
    U.append(alignend_U)


    frameCnt = np.array(f['frameCnt'])
    Vc.append(np.array(f["Vc"]))
    #assert (U[-1] == U[0]).all(), "Combining different dates with different Compositions is not yet supported"
    trial_starts.append(np.cumsum(frameCnt[:-1, 1]) + start)
    start += Vc[-1].shape[0]
print("Loaded SVD data")

trial_starts = np.concatenate( trial_starts )
Vc = np.concatenate( Vc )
U = U[0]



svd = DecompData( sessions, Vc, U, trial_starts)
svd.save( snakemake.output[0] )

snakemake_tools.stop_timer(timer_start, f"{snakemake.rule}")
