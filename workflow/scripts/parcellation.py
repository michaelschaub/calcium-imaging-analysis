from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib import DecompData

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
try:
    snakemake_tools.check_conf(snakemake, sections=["entry"])
    snakemake_tools.save_conf(snakemake, sections=["entry","parcellation"])
    start = snakemake_tools.start_timer()

    def anatom(params):
        from ci_lib.decomposition import anatomical_parcellation
        svd = DecompData.load(snakemake.input[0])
        anatomical = anatomical_parcellation(svd, atlas_path=snakemake.input["atlas"], logger=logger, **params)
        anatomical.save(snakemake.output[0])

    def locaNMF(params):
        from ci_lib.decomposition import locaNMF
        svd = DecompData.load(snakemake.input[0])
        locanmf = locaNMF(svd, atlas_path=snakemake.input["atlas"], logger=logger, **params)
        locanmf.save(snakemake.output[0])

    def ICA(params):
        from ci_lib.decomposition import fastICA
        svd = DecompData.load(snakemake.input[0], logger=logger)
        ica = fastICA(svd, **params) #snakemake.config
        ica.save(snakemake.output[0])


    parcellation = {'anatomical': anatom,
                    'ICA':ICA,
                    'LocaNMF': locaNMF}
    params = snakemake.params["params"]
    parcellation[params.pop('branch')]( params )

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
