from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent.parent).absolute()))

import numpy as np
import scipy

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib.data import DecompData
from ci_lib.decomposition import svd

import os

### Setup
logger = start_log(snakemake) # redirect std_out to log file
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["loading"])
    start = snakemake_tools.start_timer()

    params = snakemake.params["params"]
    if 'branch' in params:
        params.pop('branch')


    mask = np.logical_not(scipy.io.loadmat(snakemake.input["allenMask"])['allenMask'])

    # Load data and apply blockwise_svd
    data = DecompData.load(snakemake.input[0], logger=logger)
    np.nan_to_num(data.temporals_flat, copy=False)
    np.nan_to_num(data.spatials, copy=False)

    width = min(data.n_xaxis, mask.shape[0])
    height = min(data.n_yaxis, mask.shape[1])
    mask_sized = np.zeros((data.n_xaxis, data.n_yaxis), dtype=bool)
    mask_sized[:width,:height] = mask[:width,:height]

    data = svd(data, mask=mask_sized, **params)
    data.frame['parcellation'] = "SVD"

    ### Save
    data.save(snakemake.output[0])

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
