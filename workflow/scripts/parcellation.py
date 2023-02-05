from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData

import os

### Setup
logger = start_log(snakemake) # redirect std_out to log file
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.check_conf(snakemake, sections=[])
    snakemake_tools.save_conf(snakemake, sections=["parcellations"])
    start = snakemake_tools.start_timer()

    def anatom(params):
        from ci_lib.decomposition import anatomical_parcellation
        ### Load
        svd = DecompData.load(snakemake.input[0])
        ### Process
        anatomical = anatomical_parcellation(svd, atlas_path=snakemake.input["atlas"], logger=logger, **params)
        ### Save 
        anatomical.save(snakemake.output[0])

    def locaNMF(params):
        from ci_lib.decomposition.locanmf import locaNMF #TODO use public version 
        os.environ['NUMEXPR_MAX_THREADS'] = str(snakemake.threads)
        svd = DecompData.load(snakemake.input[0])
        locanmf = locaNMF(svd, atlas_path=snakemake.input["atlas"], logger=logger, **params)
        locanmf.save(snakemake.output[0])

    def ICA(params):
        from ci_lib.decomposition import fastICA
        svd = DecompData.load(snakemake.input[0], logger=logger)
        ica = fastICA(svd, **params) #snakemake.config
        ica.save(snakemake.output[0])

    def SVD(params):
        from ci_lib.decomposition import postprocess_SVD
        svd = DecompData.load(snakemake.input[0], logger=logger)
        svd = postprocess_SVD(svd, **params) #snakemake.config
        svd.save(snakemake.output[0])


    parcellation = {'anatomical': anatom,
                    'ICA':ICA,
                    'LocaNMF': locaNMF,
                    'SVD' : SVD}
    params = snakemake.params["params"]
    parcellation[params.pop('branch')]( params )

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
