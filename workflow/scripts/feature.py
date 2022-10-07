from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))


from ci_lib.utils import snakemake_tools
from ci_lib import DecompData
from ci_lib.features import Means, Raws, Covariances, Correlations, AutoCovariances, AutoCorrelations, Moup

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.check_conf(snakemake, sections=["parcellations","selected_trials","conditions"])
    snakemake_tools.save_conf(snakemake, sections=["parcellations","selected_trials","conditions","features"])
    start = snakemake_tools.start_timer()

    feature_dict = { "mean" : Means, "raw" : Raws, "covariance" : Covariances, "correlation" : Correlations, "autocovariance" : AutoCovariances, "autocorrelation" : AutoCorrelations, "moup" :Moup }

    feature = snakemake.params["params"]['branch']
    params = snakemake.params["params"]
    max_comps = params["max_components"] if "max_components" in params else None
    params.pop("branch",None)
    params.pop("max_components",None)

    data = DecompData.load(snakemake.input[0])
    feat = feature_dict[feature].create(data, max_comps=max_comps, **params)
    logger.debug(f"feature shape {feat.feature.shape}")

    feat.save(snakemake.output[0])

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
