import numpy as np
import h5py
import scipy
import warnings
from snakemake.logging import logger
import pandas as pd

from sklearn.decomposition import TruncatedSVD

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib.loading import load_task_data_as_pandas_df, alignment #import extract_session_data_and_save
from ci_lib import DecompData

### Setup
logger = start_log(snakemake) # redirect std_out to log file
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["parcellations"]) #fixed a bug as we dont apply parcellation to SVD and then prefilter fails to compare config as it won't contain parcellation
    timer_start = snakemake_tools.start_timer()

    ### Load
    files_Vc = snakemake.input["Vc"]
    trans_paths = snakemake.input["trans_params"]
    files_sessions = snakemake.params[0]["sessions_structured"] #snakemake.input["tasks"]

    initial_call = True
    for subject_id, fs in files_sessions.items():
        for date_time, f in fs.items():
            session = scipy.io.loadmat(f, simplify_cells=True)["SessionData"]
            logger.debug(f"session.keys {session.keys()}")
            nTrials = session["nTrials"]
            task_data = {"mouse_id": subject_id, "date_time": date_time}
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
    logger.info(total_nans)
    logger.info("Loaded task data")

    ### Process
    trial_starts = []
    Vc = []
    U = []
    start = 0
    for file_Vc,trans_path in zip(files_Vc,trans_paths):
        logger.info(f"Loaded path {file_Vc}")
        f = h5py.File(file_Vc, 'r')
        alignend_U, align_plot = alignment.align_spatials_path(np.nan_to_num(np.array(f["U"]).swapaxes(1,2)),trans_path,plot_alignment_path=snakemake.output["align_plot"])
        U.append(alignend_U) #TODO brains are already aligned in some cases (np.array(f["U"]))

        Vc.append(np.nan_to_num(np.array(f["Vc"])))
        logger.debug(f"{U[-1].shape}")
        logger.debug(f"{Vc[-1].shape}")
        n_trials, frames, n_components = Vc[-1].shape
        _, width, height = U[-1].shape
        frameCnt = frames * np.ones((n_trials),dtype=int)
        logger.info(
            f"Dimensions: n_trials={n_trials}, frames per trial={frames}, n_components={n_components}, width={width}, height={height}")
        frameCnt[0] = 0
        assert np.array_equal(U[-1].shape, U[0].shape, equal_nan=True), "Combining dates with different resolutions or number of components is not yet supported"
        

        trial_starts.append(np.cumsum(frameCnt) + start)
        logger.debug(repr(trial_starts))
        Vc[-1] = np.concatenate( Vc[-1] )
        start += Vc[-1].shape[0]

    trial_starts = np.concatenate( trial_starts )
    

    if len(U)==1:
        U = U[0]
    else:

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        flat_U = np.concatenate([u.reshape(n_components,width * height) for u in U])
        
        svd.fit(np.nan_to_num(flat_U))
        
        #mean_U = np.mean(np.nan_to_num(U),axis=0)


        #spat_n, w , h = mean_U.shape

        mean_U = svd.components_  #mean_U.reshape(spat_n,w * h)

        mean_U_inv = np.nan_to_num(np.linalg.pinv(np.nan_to_num(mean_U, nan=0.0)), nan=0.0)

        #approx_U_error = np.matmul(mean_U_inv, mean_U)

        error = np.zeros((len(U)))

        for i,V in enumerate(Vc):
            U[i] = U[i].reshape(n_components,width * height)

            V_transform = np.matmul(np.nan_to_num(U[i], nan=0.0), mean_U_inv)

            Vc[i] = np.matmul(Vc[i], V_transform)

        U = mean_U.reshape(n_components,width,height) # U[0]

    Vc = np.concatenate( Vc )

    logger.debug(f"{U.shape}")
    logger.debug(f"{Vc.shape}")

    ### Save
    print(sessions)
    svd = DecompData( sessions, Vc, U, trial_starts)
    logger.debug(f"svd dataframe:\n{str(svd._df)}")
    svd.save( snakemake.output[0] )

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
