from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import numpy as np

from ci_lib.utils import snakemake_tools
from ci_lib import DecompData
from ci_lib.features import Means, Raws, Covariances, Correlations, AutoCovariances, AutoCorrelations, Moup


logger = snakemake_tools.start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    start = snakemake_tools.start_timer()

    feature_dict = { "mean" : Means, "raw" : Raws, "covariance" : Covariances, "correlation" : Correlations, "autocovariance" : AutoCovariances, "autocorrelation" : AutoCorrelations, "moup" :Moup }
    feature_class = feature_dict[snakemake.wildcards["feature"].split("_")[0]]
    feat = feature_class.load(snakemake.input["data"])

    print(snakemake.params["params"])
    thresh_feat = feat.feature
    print(thresh_feat)
    thresh_feat[np.absolute(thresh_feat)<0.05*np.max(thresh_feat)]= 0
    print(thresh_feat)
    feat.feature = thresh_feat


    feat.save(snakemake.output["data"])

    snakemake_tools.save(snakemake, snakemake.output["export_raw"], feat.feature)


    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
