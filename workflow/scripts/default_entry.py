import numpy as np
import h5py
import warnings
from snakemake.logging import logger

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.loading import load_task_data_as_pandas_df, alignment #import extract_session_data_and_save
from ci_lib import DecompData

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["entry","parcellation"]) #fixed a bug as we dont apply parcellation to SVD and then prefilter fails to compare config as it won't contain parcellation
    timer_start = snakemake_tools.start_timer()

    files_Vc = snakemake.input["Vc"]
    task_files = snakemake.params["task_structured"] #snakemake.input["tasks"]
    trans_paths = snakemake.input["trans_params"]
    mouse_dates_str = snakemake.params["mouse_dates_str"]

    sessions = load_task_data_as_pandas_df.extract_session_data_and_save(
            root_paths=task_files, mouse_dates_str = mouse_dates_str, reextract=True, logger=logger) #reextraction needs to be done for different set of dates otherwise session will have wrong dims
    logger.info("Loaded task data")


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
        assert np.array_equal(U[-1], U[0], equal_nan=True), "Combining different dates with different Compositions is not yet supported"
        trial_starts.append(np.cumsum(frameCnt[:-1, 1]) + start)
        start += Vc[-1].shape[0]

    trial_starts = np.concatenate( trial_starts )
    Vc = np.concatenate( Vc )



    U = U[0]



    svd = DecompData( sessions, Vc, U, trial_starts)
    svd.save( snakemake.output[0] )

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
