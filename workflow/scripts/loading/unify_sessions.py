import numpy as np
import h5py
import scipy
import warnings
from snakemake.logging import logger
from sklearn.decomposition import TruncatedSVD


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

    if snakemake.params['alias']:
        input_path = Path(snakemake.input[0])
        output_path = Path(snakemake.output[0])
        base_path = input_path.parent.parent.parent.parent
        raise ValueError

        snakemake_tools.stop_timer(timer_start, logger=logger)
        sys.exit(0)

    input_files = list(snakemake.input)
    logger.debug(f"{input_files=}")

    sessions = []
    trial_starts = []
    Vc = []
    U = []
    start = 0
    for f in input_files:
        data = DecompData.load(f)

        sessions.append(data._df)
        U.append(data.spatials)
        Vc.append(data.temporals_flat)
        trial_starts.append(data._starts + start)
        start += Vc[-1].shape[0]

    sessions = pd.concat(sessions)
    trial_starts = np.concatenate( trial_starts )

    frames, n_components = Vc[-1].shape
    _, width, height = U[-1].shape
    logger.debug(f"{frames=}, {n_components=}, {width=}, {height=}")
    logger.debug(f"{len(U)=}")
    svd = TruncatedSVD(n_components=n_components, random_state=snakemake.config['seed'])
    flat_U = np.nan_to_num(np.concatenate([u.reshape(n_components,width * height) for u in U]))
    logger.debug(f"{flat_U.shape=}")
    
    svd.fit(flat_U)
    
    logger.debug(f"{svd.components_.shape=}")
    mean_U = svd.components_ 

    mean_U_inv = np.nan_to_num(np.linalg.pinv(np.nan_to_num(mean_U, nan=0.0)), nan=0.0)

    error = np.zeros((len(U)))

    for i,V in enumerate(Vc):
        U[i] = U[i].reshape(n_components,width * height)
        V_transform = np.matmul(np.nan_to_num(U[i], nan=0.0), mean_U_inv)
        Vc[i] = np.matmul(Vc[i], V_transform)

    U = mean_U.reshape(n_components,width,height) # U[0]
    Vc = np.concatenate( Vc )

    svd = DecompData( sessions, Vc, U, trial_starts, allowed_overlap=0) #TODO remove hardcode
    svd.save( snakemake.output[0] )

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
