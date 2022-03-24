from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))


from ci_lib.utils import snakemake_tools
from ci_lib import DecompData
from ci_lib.features import Means, Raws, Covariances, Correlations, AutoCovariances, AutoCorrelations, Moup

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
try:
    snakemake_tools.check_conf(snakemake, sections=["entry","parcellation","trial_selection","conditions"])
    snakemake_tools.save_conf(snakemake, sections=["entry","parcellation","trial_selection","conditions","feature_calculation"])
    start = snakemake_tools.start_timer()

    config = snakemake.config["rule_conf"]["feature_calculation"]


    feature_dict = { "mean" : Means, "raw" : Raws, "covariance" : Covariances, "correlation" : Correlations, "autocovariance" : AutoCovariances, "autocorrelation" : AutoCorrelations, "moup" :Moup }

    # Dictionary for converting parameters from workflow to parameters, that can be passed to feature creators
    param_dict = {
            "mean"              : (lambda p : {}),
            "raw"               : (lambda p : {}),
            "covariance"        : (lambda p : {}),
            "correlation"        : (lambda p : {}),
            # convert parameter "max_timelag" to range up to that timelag, if "max_timelag" does not exist, pass "timelags" (iterable)
            "autocovariance"    : (lambda p : { "timelags" : range(1,p["max_timelag"]+1) if "max_timelag" in p else p["timelags"] }),
            "autocorrelation"    : (lambda p : { "timelags" : range(1,p["max_timelag"]+1) if "max_timelag" in p else p["timelags"] }),
            # no conversion needed for Moup
            "moup"              : (lambda p : {"timelag": p["timelags"]})}

    feature = snakemake.params["params"]['branch']
    params = snakemake.params["params"]
    data = DecompData.load(snakemake.input[0])

    max_comps = params["max_components"] if "max_components" in params else None
    feat = feature_dict[feature].create(data, max_comps=max_comps, **param_dict[feature](params))
    logger.debug(f"feature shape {feat._feature.shape}")

    feat.save(snakemake.output[0])

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
