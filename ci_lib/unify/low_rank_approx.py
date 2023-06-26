from pathlib import Path
import scipy.io
import numpy as np
from sklearn.decomposition import FastICA
import wlra
from sklearn.decomposition import TruncatedSVD

import logging
LOGGER = logging.getLogger(__name__)


from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib.loading import load_task_data_as_pandas_df, alignment #import extract_session_data_and_save
from ci_lib.plotting import draw_neural_activity
from ci_lib import DecompData

def weighted_lra(VCs,Us,sessions, trial_starts, seed=0):
    """Computes a shared space (low rank approximation) across spaces of different sessions

    Args:
        VCs (_type_): _description_
        Us (_type_): _description_
        sessions (_type_): _description_
        trial_starts (_type_): _description_
    """
    frames, n_components = VCs[-1].shape
    _, width, height = Us[-1].shape
    LOGGER.debug(f"{frames=}, {n_components=}, {width=}, {height=}")
    LOGGER.debug(f"{len(Us)=}")

    ##SVD
    #svd = TruncatedSVD(n_components=n_components, random_state=snakemake.config['seed'])

    flat_U = np.nan_to_num(np.concatenate([u.reshape(n_components,width * height) for u in Us]))
    flat_VC_sum = np.nan_to_num(np.concatenate([np.repeat(np.sum(vc,axis=0)[:,np.newaxis],width * height,axis=1) for vc in VCs])) #Compute activiation of spatial component across all frames, repeat weight for each pixel of map #TODO exclude pixels outside of brain map
    flat_VC_sum /= np.amax(flat_VC_sum) #normalize for WLRA

    LOGGER.debug(f"{flat_U.shape=}")
    
    #svd.fit(flat_U)
    shared_U = wlra.wlra(flat_U, flat_VC_sum , rank=n_components, max_iters=1000, verbose=True)

    LOGGER.debug(f"{shared_U.shape=}")
    shared_U = shared_U[:n_components,:] #only use first n_components
    
    LOGGER.debug(f"{shared_U.shape=}")
    #mean_U = svd.components_ 

    mean_U_inv = np.nan_to_num(np.linalg.pinv(np.nan_to_num(mean_U, nan=0.0)), nan=0.0)

    error = np.zeros((len(Us)))

    for i,V in enumerate(VCs):
        Us[i] = Us[i].reshape(n_components,width * height)
        V_transform = np.matmul(np.nan_to_num(Us[i], nan=0.0), mean_U_inv)
        VCs[i] = np.matmul(VCs[i], V_transform)

    U = mean_U.reshape(n_components,width,height) # U[0]
    Vc = np.concatenate( VCs )

    return DecompData( sessions, Vc, U, trial_starts, allowed_overlap=0)

def lra(VCs,Us,sessions, trial_starts, seed=0):
    """Computes a shared space (low rank approximation) across spaces of different sessions

    Args:
        VCs (_type_): _description_
        Us (_type_): _description_
        sessions (_type_): _description_
        trial_starts (_type_): _description_
    """
    frames, n_components = VCs[-1].shape
    _, width, height = Us[-1].shape
    LOGGER.debug(f"{frames=}, {n_components=}, {width=}, {height=}")
    LOGGER.debug(f"{len(Us)=}")

    ##SVD
    svd = TruncatedSVD(n_components=n_components, random_state=seed)

    flat_U = np.nan_to_num(np.concatenate([u.reshape(n_components,width * height) for u in Us]))

    LOGGER.debug(f"{flat_U.shape=}")
    
    svd.fit(flat_U)
    
    LOGGER.debug(f"{svd.components_.shape=}")
    mean_U = svd.components_ 

    mean_U_inv = np.nan_to_num(np.linalg.pinv(np.nan_to_num(mean_U, nan=0.0)), nan=0.0)


    for i,V in enumerate(VCs):
        Us[i] = Us[i].reshape(n_components,width * height)
        V_transform = np.matmul(np.nan_to_num(Us[i], nan=0.0), mean_U_inv)
        VCs[i] = np.matmul(VCs[i], V_transform)

    U = mean_U.reshape(n_components,width,height) # U[0]
    Vc = np.concatenate( VCs )

    return DecompData( sessions, Vc, U, trial_starts, allowed_overlap=0)
