from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import numpy as np

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib.plotting import plot_connectivity_matrix
from ci_lib import DecompData
from ci_lib.features import from_string as feat_from_string

### Setup
logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    start = snakemake_tools.start_timer()

    ### Load
    feature_class = feat_from_string(snakemake.wildcards["feature"].split("_")[0])
    feat = feature_class.load(snakemake.input["data"])

    threshold = snakemake.params["params"]["thresh"]

    ### Process
    thresh_feat = np.array(feat.feature)
    thresh_feat[np.absolute(thresh_feat)<threshold*np.mean(np.absolute(thresh_feat))]= 0


    ### Save
    plot_connectivity_matrix([np.mean(feat.feature,axis=0)[0],np.mean(thresh_feat,axis=0)[0]],title=snakemake.wildcards["feature"],path=snakemake.output["export_plot"]) #TODO why is it trial x 1 (?) x w x h

    feat.feature = thresh_feat
    snakemake_tools.save(snakemake.output["export_raw"], feat.feature)

    feat.save(snakemake.output["data"])



    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
