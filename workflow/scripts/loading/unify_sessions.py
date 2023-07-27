import numpy as np
from sklearn.decomposition import TruncatedSVD

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData

import pandas as pd

# redirect std_out to log file
logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["parcellations"]) #fixed a bug as we dont apply parcellation to SVD and then prefilter fails to compare config as it won't contain parcellation
    timer_start = snakemake_tools.start_timer()

    ### If alias is set, no calculation is needed
    if snakemake.params['alias']:
        input_path = Path(snakemake.input[0])
        output_path = Path(snakemake.output[0])
        base_path = input_path.parent.parent.parent.parent
        raise NotImplemented("Aliasing not yet implemented")

        snakemake_tools.stop_timer(timer_start, logger=logger)
        sys.exit(0)

    ### Load individual DecompData objects

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

    ### Begin SVD

    frames, n_components = Vc[-1].shape
    _, width, height = U[-1].shape
    logger.debug(f"{frames=}, {n_components=}, {width=}, {height=}")
    logger.debug(f"{len(U)=}")
    svd = TruncatedSVD(n_components=n_components, random_state=snakemake.config['seed'])
    # flatten U's width and height, concatenate along components and replace nans with zero
    flat_U = np.nan_to_num(np.concatenate([u.reshape(n_components,width * height) for u in U]))
    logger.debug(f"{flat_U.shape=}")
    
    # actual SVD calulation
    svd.fit(flat_U)
    
    # new U, still flattened
    logger.debug(f"{svd.components_.shape=}")
    mean_U = svd.components_ 

    # the pseudo inverse is required to calculate new temporals
    mean_U_inv = np.nan_to_num(np.linalg.pinv(np.nan_to_num(mean_U, nan=0.0)), nan=0.0)

    error = np.zeros((len(U)))

    for i,V in enumerate(Vc):
        U[i] = U[i].reshape(n_components,width * height)
        # calculate tranformation matrix U * mean_U^(-1) and use it to transform Vc,
        # more efficient then calculating (Vc * U) * mean_U^(-1)
        V_transform = np.matmul(np.nan_to_num(U[i], nan=0.0), mean_U_inv)
        # transfor temporals
        Vc[i] = np.matmul(Vc[i], V_transform)

    # reshape into new spatials
    U = mean_U.reshape(n_components,width,height) # U[0]
    # concatenate temporals trials into one temporals
    Vc = np.concatenate( Vc )

    ### Save into new DecompData

    svd = DecompData( sessions, Vc, U, trial_starts, allowed_overlap=0) #TODO remove hardcode
    svd.frame['decomposition_space'] = snakemake.wildcards['dataset_id']
    svd.save( snakemake.output[0] )

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
