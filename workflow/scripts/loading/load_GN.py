import numpy as np
import h5py
import scipy
import warnings
from snakemake.logging import logger

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib.loading import load_task_data_as_pandas_df, alignment #import extract_session_data_and_save
from ci_lib.plotting import draw_neural_activity
from ci_lib import DecompData

import pandas as pd
import seaborn as sns


# redirect std_out to log file
logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["parcellations"]) #fixed a bug as we dont apply parcellation to SVD and then prefilter fails to compare config as it won't contain parcellation
    timer_start = snakemake_tools.start_timer()

    file_Vc = snakemake.input["Vc"]
    task_files = snakemake.input["tasks"]
    trans_path = snakemake.input["trans_params"]
    task_paths = { snakemake.wildcards['session_id'].split('-')[0] : [task_files] }

    sessions = load_task_data_as_pandas_df.extract_session_data_and_save(
            root_paths=task_paths, reextract=True, logger=logger) #reextraction needs to be done for different set of dates otherwise session will have wrong dims

    logger.info("Loaded sessions. Processing...")

    sessions['subject_id'] = sessions['mouse_id']

    #annotate meta data with switch/non switch from previous stimulus and mouse choice to balance trials (with out requiring their order)
    #1: changed, 0: no change, -1: unknown (due to being the first task)
    response_side_diff = sessions['response_left'].diff().fillna(1).astype('int')
    response_side_diff[response_side_diff>0] = 1
    sessions['change_response_side'] = response_side_diff

    stim_side_diff = sessions['target_side_left'].diff().fillna(1).astype('int')
    stim_side_diff[stim_side_diff>0] = 1
    sessions['change_stim_side'] = stim_side_diff
    
    



    logger.info(f"{sessions=}")
    logger.info("Done.")

    ###
    switch  = []
    same = []

    for i,trial in enumerate((sessions['target_side_left'])):
        if i>0:
            if prev_trial == trial:
                same.append(trial)
            else:
                switch.append(trial)
        prev_trial = trial
    same = np.asarray(same)
    switch = np.asarray(switch)

    left_switch = len(switch[switch==1])
    right_switch = len(switch[switch==0])
    right_same = len(same[same==0])
    left_same = len(same[same==1])

    sns.set()
    ax = sns.barplot(x=["left_switch","right_switch","left_same","right_same"],y=[left_switch,right_switch,left_same,right_same])


        #ax = sns.heatmap( pd.DataFrame(sessions['target_side_left'][:100]))
    ax.figure.savefig(snakemake.output["stim_side"])

    #ax = sns.heatmap( pd.DataFrame(sessions['target_side_left'][:100]))
    #ax.figure.savefig(snakemake.output["stim_side"])


    logger.info("Loaded task data")


    #if len(files_Vc) > 1:
    #    warnings.warn("Combining different dates may still be buggy!")

    print(file_Vc)
    try:
        f = h5py.File(file_Vc, 'r')
    except OSError as err:
        if "Unable to open file (file signature not found)" in str(err):
            f = scipy.io.loadmat(file_Vc)
            warnings.warn("You are using an old mat format (<Matlab 7.3 from 2015), please switch to HDF5 to improve perfromance")
            for key,data in f.items():
                if isinstance(data, (list, tuple, np.ndarray)):
                    f[key] = np.transpose(np.squeeze(data))

        else:
            raise Exception("Your mat file has problems... Please ensure it is not corrupted")
            

    #Aligns spatials for each date with respective trans_params
    U, align_plot = alignment.align_spatials_path(np.array(f["U"]),trans_path,plot_alignment_path=snakemake.output["align_plot"])

    frameCnt = np.array(f['frameCnt'])
    Vc = np.array(f["Vc"])

    frames, n_components = Vc.shape
    _, width, height = U.shape
    logger.info(f"{frameCnt.shape}")
    logger.info(
        f"Dimensions: n_trials={len(frameCnt)-2}, average frames per trial={frames/(len(frameCnt)-2)}, n_components={n_components}, width={width}, height={height}")

    trial_starts = np.cumsum(frameCnt[:-1, 1])

    #####
    if "pretrial" in snakemake.config["phase_conditions"]:
        overlap = int(snakemake.config["phase"]["pretrial"]["stop"]) - int(snakemake.config["phase"]["pretrial"]["start"]) #pretrial of following trial is attached to posttrial of previous trial
    else:
        overlap = 0

    svd = DecompData( sessions, Vc, U, trial_starts, allowed_overlap=0) #TODO remove hardcode
    svd.frame['parcellation'] = 'SVD'
    svd.frame['decomposition_set_id'] = snakemake.wildcards['session_id']
    svd.frame[svd.dataset_id_column] = snakemake.wildcards['session_id']
    svd.frame['session_id'] = snakemake.wildcards['session_id']
    svd.save( snakemake.output[0] )

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
