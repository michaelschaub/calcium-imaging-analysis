from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))


from ci_lib.utils import snakemake_tools
from ci_lib import DecompData
from ci_lib.features import Means, Raws, Covariances, AutoCovariances, Moup

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
try:
    snakemake_tools.check_conf(snakemake, sections=["entry","parcellation","trial_selection","conditions"])
    snakemake_tools.save_conf(snakemake, sections=["entry","parcellation","trial_selection","conditions","feature_calculation"])
    start = snakemake_tools.start_timer()

    config = snakemake.config["rule_conf"]["feature_calculation"]


    feature_dict = { "mean" : Means, "raw" : Raws, "covariance" : Covariances, "autocovariance" : AutoCovariances, "moup" :Moup }

    param_dict = {
            "mean"              : (lambda p : {}),
            "raw"               : (lambda p : {}),
            "covariance"        : (lambda p : {}),
            "autocovariance"    : (lambda p : { "time_lag_range" : range(1,p["max_timelag"]+1) if "max_timelag" in p else p["timelags"] }),
            "moup"              : (lambda p : {"timelag": p["timelags"]})}

    feature = snakemake.wildcards["feature"].split("_")[0]
    params = snakemake.params["params"]
    data = DecompData.load(snakemake.input[0])

    feat = feature_dict[feature].create(data, max_comps=params["max_components"], **param_dict[feature](params))


    feat.save(snakemake.output[0])

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
