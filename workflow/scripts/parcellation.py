from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib import DecompData
from ci_lib.decomposition import anatomical_parcellation, fastICA, locaNMF as lnmf

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
try:
    snakemake_tools.check_conf(snakemake, sections=["entry"])
    snakemake_tools.save_conf(snakemake, sections=["entry","parcellation"])
    start = snakemake_tools.start_timer()

    def anatom():
        svd = DecompData.load(snakemake.input[0])
        anatomical = anatomical_parcellation(svd, dict_path=snakemake.input["meta"])
        anatomical.save(snakemake.output[0])

    def locaNMF():
        svd = DecompData.load(snakemake.input[0])
        locanmf = lnmf(svd, atlas_path=snakemake.input["atlas"])
        locanmf.save(snakemake.output[0])

    def ICA():
        svd = DecompData.load(snakemake.input[0])
        ica = fastICA(svd, 64) #snakemake.config
        ica.save(snakemake.output[0])


    parcellation = {'anatomical': anatom,
                    'ICA':ICA,
                    'LocaNMF': locaNMF}
    parcellation[snakemake.wildcards['parcellation']]()

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
