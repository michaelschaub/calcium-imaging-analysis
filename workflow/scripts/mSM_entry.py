import numpy as np
import h5py
import scipy
import warnings
from snakemake.logging import logger
import pandas as pd

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.loading import load_task_data_as_pandas_df, alignment #import extract_session_data_and_save
from ci_lib import DecompData

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["entry","parcellation"]) #fixed a bug as we dont apply parcellation to SVD and then prefilter fails to compare config as it won't contain parcellation
    timer_start = snakemake_tools.start_timer()

    files_Vc = snakemake.input["Vc"]
    files_sessions = snakemake.params["sessions_structured"] #snakemake.input["tasks"]
    mouse_dates_str = snakemake.params["mouse_dates_str"]

    initial_call = True
    for mouse_id, fs in files_sessions.items():
        for date_time, f in fs.items():
            session = scipy.io.loadmat(f, simplify_cells=True)["SessionData"]
            logger.debug(f"session.keys {session.keys()}")
            nTrials = session["nTrials"]
            task_data = {"mouse_id": mouse_id, "date_time": date_time}
            for k, data in session.items():
                logger.debug(f"{k}: {type(data)}")
                if isinstance(data, np.ndarray):
                    logger.debug(f"{k}: {data.dtype} {data.shape}")
                    if data.shape == (nTrials,):
                        logger.info(f"Included {k} in sessions, matches shape (nTrials,)")
                        task_data[k] = data
                elif isinstance(data, dict):
                    logger.debug(f"{k}: {[ (k, type(d)) for k,d in data.items()]}")
                elif isinstance(data, list):
                    logger.debug(f"{k}: {len(data)} x {data[0]}")
                else:
                    logger.debug(f"{k}: {data}")

            if initial_call:
                sessions = pd.DataFrame.from_dict(task_data)
                initial_call = False
            else:
                sessions = sessions.append(pd.DataFrame.from_dict(task_data), ignore_index=True)

    total_nans = []
    for k in sessions.keys():
        try:
            a = np.array( sessions[k], dtype=float)
            nans = np.nonzero(np.isnan(a))[0]
            if nans.shape != (0,):
                logger.debug(f"{k}: {nans}")
            else:
                logger.debug(f"{k}: no nans")
            total_nans.extend( nans )
        except ValueError:
            logger.debug(f"{k}: could not convert to float, no nan test")
    sessions = sessions.drop(sessions.index[total_nans])
    logger.info(f"Dropped {len(total_nans)} trials, because of NaN entries.")

        
    logger.info("Loaded task data")


    if len(files_Vc) > 1:
        warnings.warn("Combining different dates may still be buggy!")

    trial_starts = []
    Vc = []
    U = []
    start = 0
    for file_Vc in files_Vc:
        f = h5py.File(file_Vc, 'r')

        U.append(np.array(f["U"]).swapaxes(1,2))

        Vc.append(np.array(f["Vc"]))
        logger.debug(f"{U[-1].shape}")
        logger.debug(f"{Vc[-1].shape}")
        n_trials, frames, n_components = Vc[-1].shape
        _, width, height = U[-1].shape
        frameCnt = frames * np.ones((n_trials),dtype=int)
        logger.info(
            f"Dimensions: n_trials={n_trials}, frames per trial={frames}, n_components={n_components}, width={width}, height={height}")
        frameCnt[0] = 0
        assert np.array_equal(U[-1], U[0], equal_nan=True), "Combining different dates with different Compositions is not yet supported"
        trial_starts.append(np.cumsum(frameCnt) + start)
        logger.debug(repr(trial_starts))
        Vc[-1] = np.concatenate( Vc[-1] )
        start += Vc[-1].shape[0]

    trial_starts = np.concatenate( trial_starts )
    Vc = np.concatenate( Vc )
    U = U[0]
    logger.debug(f"{U.shape}")
    logger.debug(f"{Vc.shape}")



    svd = DecompData( sessions, Vc, U, trial_starts)
    logger.debug(f"svd dataframe:\n{str(svd._df)}")
    svd.save( snakemake.output[0] )

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
