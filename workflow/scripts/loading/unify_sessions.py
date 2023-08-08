import numpy as np
from sklearn.decomposition import TruncatedSVD
import scipy

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData
from ci_lib.decomposition import blockwise_svd

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

    method = snakemake.params['method']

    ### Load individual DecompData objects

    input_files = list(snakemake.input)
    logger.debug(f"{input_files=}")

    logger.info(f"Starting unification with method {method}")
    if method in ("sv_weighted", "naiv"):
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

            if method == "sv_weighted":
                # rescale U by singular values, the inverse with Vc
                # Does not need to be reversed later, since Vc are transformed with U in the end anyway
                singular_values = np.linalg.norm(Vc[-1], axis=0)
                U[-1] *= singular_values[:, None, None]
                Vc[-1] /= singular_values[None, :]

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
            # transform temporals
            Vc[i] = np.matmul(Vc[i], V_transform)

        # reshape into new spatials
        U = mean_U.reshape(n_components,width,height) # U[0]
        # concatenate temporals trials into one temporals
        Vc = np.concatenate( Vc )
    elif method == "block_svd":
        all_data = [ DecompData.load(f) for f in input_files ]

        mask_path = snakemake.config['paths']['parcellations'].get('SVD',{}).get('allenMask', None)
        if mask_path is None:
            logger.warning(f"No Mask found!")
            mask = None
        else:
            # convert from mask, so that True indicates brain, not not brain
            mask = np.logical_not(scipy.io.loadmat(mask_path)['allenMask'])
            width = min(all_data[-1].n_xaxis, mask.shape[0])
            height = min(all_data[-1].n_yaxis, mask.shape[1])
            mask_sized = np.zeros((all_data[-1].n_xaxis, all_data[-1].n_yaxis), dtype=bool)
            mask_sized[:width,:height] = mask[:width,:height]
            mask = mask_sized
        for data in all_data:
            # remove nans in all components
            np.nan_to_num(data.temporals_flat, copy=False)
            np.nan_to_num(data.spatials, copy=False)
        # create pixel object spanning all sessions
        pixel = DecompData.PixelSlice.concat([data.pixel for data in all_data])
        # actual blockwise svd
        # TODO remove hardcoded blocksize
        Vc, U, S = blockwise_svd(pixel, all_data[-1].n_components, blocksize=60, logger=logger, mask=mask)
        logger.debug(f"SVD spatial nans: {np.where(np.isnan(U))}")

        # create new trial_starts and sessions dataframe
        trial_starts = []
        start = 0
        for data in all_data:
            trial_starts.append(data._starts + start)
            start += data.t_max
        trial_starts = np.concatenate( trial_starts )
        sessions = pd.concat([data.frame for data in all_data])
    else:
        raise ValueError(f"Unknown unification method '{method}'")

    ### Save into new DecompData

    svd = DecompData( sessions, Vc, U, trial_starts, allowed_overlap=0) #TODO remove hardcode
    logger.debug(f"DecompData spatial nans: {np.where(np.isnan(svd.spatials))}")

    svd.frame['decomposition_space'] = snakemake.wildcards['dataset_id']
    svd.frame['unification_method']  = method
    svd.save( snakemake.output[0] )

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
